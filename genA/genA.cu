/**
   Genetic algorithm optimizer
   genA.cu
   Runs iterations of a genetic algoirthm to optimize molecular mechanics dihedral parameters

   @author James Maier
   @version 1.0 2014 Jul 29
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <math.h>
#include <fstream>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/generate.h>
#include <thrust/device_ptr.h>
/*#undef __GLIBCXX_ATOMIC_BUILTINS
#undef __GLIBCXX_USE_INT128
#define _GLIBCXX_GTHREAD_USE_WEAK 0 */
#include <list>
#include <map>
#include "load.cpp"
 
  const int BLOCK_SIZE=256;

//#define HANDLE_ERROR(x) x;error=cudaGetLastError();if(error!=cudaSuccess){printf("CUDA error: %s\n", cudaGetErrorString(error));exit(-1);}

#define HANDLE_ERROR(x) x;

/** 
 *  creates offspring from a population, generating crossovers according to pCross
 *
 *  @param Vs a global array of all the parent and child genomes
 *  @param ptrs array of pointers from logical indices to actual indices into Vs for each individual
 *  @param areas the probabilities for choosing each individual for mating
 *  @param sumArea pointer to the sum of all the individual areas
 *  @param rands array of random numbers
 *  @param pCross probability that crossover occurs
 *  @param pSize number of individuals in the population
 *  @param genomeSize number of genes in a genome
 */
__global__ void mateIt(float *Vs, int *ptrs, const float *areas, const float *sumArea, const float *rands, const float pCross, const int pSize, const int genomeSize)
{
  /* figure out index */
  int i=blockIdx.x * blockDim.x + threadIdx.x;
  /* first parent, second parent, crossover random numbers */
  int randi=i*3;
  // multiply i by 2, as we will have 2 parents and 2 offspring
  i<<=1;
  // if we're in the population (sometimes warps may go past)
  if(i<pSize){
    int parent[2];
    int j, from=0;
    /* figure out parents */
    parent[0]=parent[1]=-1;
    // find parent where cumulative (cum) area (A) is less than random target (tgt) area
    float cumA=0.0f, tgtA=rands[randi++]* *sumArea;
    while(cumA<=tgtA){
      ++parent[0];
      cumA+=areas[ptrs[parent[0]]/genomeSize];
    }
  #if DEBUG>2
    printf("rands[%d] ; %f ; %f=%f * %f\n",randi, cumA, tgtA, rands[randi-1], *sumArea);
  #endif
    cumA=0.0f; tgtA=rands[randi++]*
          (*sumArea-areas[ptrs[parent[0]]/genomeSize]);
    while(cumA<=tgtA){
      ++parent[1];
      if(parent[1]==parent[0])
        ++parent[1];
      cumA+=areas[ptrs[parent[1]]/genomeSize];
    }
  #if DEBUG>2
    printf("Make offspring %d from %d and %d (%f=%f*(%f-%f)) %d\n", i, p[0], p[1], tgtA, rands[randi-1], *sumArea, areas[ptrs[p[0]]/genomeSize], randi);
  #endif
    /* add offset of pSize to i because it is a child (next population) */
    i+=pSize;
    /* use ptrs to get indices into Vs */
    int i0=ptrs[i], i1=ptrs[i+1];
    parent[0]=ptrs[parent[0]];
    parent[1]=ptrs[parent[1]];
    /* set j to index for the next set of Vs */
    j=i0+genomeSize;
    /* put parent[0], parent[1], and i1 relative to i0, so we can just add i0 for index */
    parent[0]-=i0;
    parent[1]-=i0;
    i1-=i0;
    /* start with crossover pt at the end (no crossover) */
    int crossPt=j;
    /* check for crossover */
    if(rands[randi]<pCross){
      crossPt=i0+1+(int)(rands[randi]/pCross*(float)(genomeSize-1));
    }
    while(i0<crossPt){
      /* load next bit from parent and increment i */
      Vs[i0]=Vs[parent[0]+i0];
      Vs[i1+i0]=Vs[parent[1]+i0];
      ++i0;
    }
    while(i0<j){
      Vs[i0]=Vs[parent[1]+i0];
      Vs[i1+i0]=Vs[parent[0]+i0];
      ++i0;
    }
  }
}

/**
 * @brief introduces mutations to the genomes in Vs, according to probability pMut, with a max perturbation of max
 *
 * @param Vs a global array of all the parent and child genomes
 * @param ptrs array of pointers from logical indices to actual indices into Vs for each individual
   @param rands array of random numbers
 * @param pSize number of individuals in the population
 * @param pMut probability that a mutation occurs, evaluated for each gene
 * @param max maximum perturbation to an allele
 * @param genomeSize number of genes in a genome
 */
__global__ void mutateIt(float *Vs, int *ptrs, const float *rands, const int pSize, const float pMut, const float max, const int genomeSize)
{
  /* figure out index */
  int i=blockIdx.x * blockDim.x + threadIdx.x;
  if(i<pSize){
    // get index into random number array
    int r=i*genomeSize;
    i=ptrs[i];
    int j=i+genomeSize;
    // want random numbers from [-max, max). will subtract max later
    float scale=2.0f*max/pMut;
    // iterate through genome
    while(i<j){
      if(rands[r]<pMut){
        // mutate the amplitude by adding perturbation
        Vs[i]+=rands[r]*scale-max;
      }
      ++i;
      ++r;
    }
  }
}

/**
 * @brief calculates a score indicating the closeness of fit for each individual (set of parameters) against the training set
 *
 * @param scores score for each conformation, calculated here
 * @param areas weighting for each conformation, was formerly calculated here
 * @param Vs a global array of all the parent and child genomes
 * @param ptrs array of pointers from logical indices to actual indices into Vs for each individual
 * @param tset training set
 * @param tgts targets for training
 * @param wts weights of each point in the training set
 * @param breaks breaks in training set, where different data should not be compared across breaks
 * @param nConf number of conformations in training set
 * @param pSize number of individuals in the population
 * @param genomeSize number of genes in a genome
 * @param xx space to store energy differences for each conformation with test parameters
 */
__global__ void scoreIt(float *scores, float *areas, const float *Vs, const int *ptrs, const float *tset, const float *tgts, const float *wts, const int *breaks, const int nConf, const int pSize, const int genomeSize, float *xx)
{
  int i=blockIdx.x * blockDim.x + threadIdx.x;
  //if((i<<1)<(pSize-1)*pSize){
  if(i<pSize){
  float *x=xx+i*nConf;  // for the error of each conformation
  // don't add pSize to get into children
  // i+=pSize;
  // get reference to score
  float *S=scores+i;
  // set score to 0
  *S=0.0f;
  // accumulate little s for each set
  float s;
  // get first index in genome
  int i0=ptrs[i];
  // get index of next genome space for looping bounds
  int j=i0+genomeSize;
  // start with the first element in the training set
  int t=0;
  /* start at break 0 */
  int b=0;
  /* loop over conformations c */
  int c=0;
  while(c<nConf){
    //int nP=0;
    s=0.0f;
    /* loop only in units without break points */
    while(c<breaks[b+1]){
      /* start with error = energy difference of conformation c */
      //printf("x[%d]\n",c);
      x[c]=tgts[c];
      /* subtract contributions from each parameter for conformation c */
      for(i=i0;i<j;i++,t++){
        x[c]-=tset[t]*Vs[i];
      }
      /* add differences in this error from all other errors */
      for(int c2=breaks[b];c2<c;c2++){
        float err=x[c]-x[c2];
        s+=(err<0.0f?-err:err);
        //++nP;
      }
      /* next conformation */
      ++c;
    }
    /* add little error to big error S, weighted by number of pairs */
    *S+=s*wts[b];
    /* go to next breakpoint */
    ++b;
  }
  //areas[i0/genomeSize]=__expf(-*S/*formerBest);
#if DEBUG>1
  printf("areas[%d]=%f\n",i0/genomeSize,areas[i0/genomeSize]);
#endif
  }
}

/**
 * @brief calculates the areas -- the probability each individual has of mating
 *
 * @param scores scores for each individual (set of parameters)
 * @param areas fitness for each individual, in terms of probability of mating
 * @param ptrs array of pointers from logical indices to actual indices into Vs for each individual
 * @param pSize number of individuals in the population
 * @param genomeSize number of genes in a genome
*/
__global__ void calcAreas(float *scores, float *areas, const int *ptrs, const int pSize, const int genomeSize) {
  int i=blockIdx.x * blockDim.x + threadIdx.x;
  //if((i<<1)<(pSize-1)*pSize){
  if(i<pSize){
    areas[ptrs[i]/genomeSize]=__expf(-scores[i]/scores[0]);
  }
}

/**
* @brief simple helper function for copying data from oldF, oldI to neWF, newI
*
* @param newF pointer to new float array
* @param newI pointer to new int array
* @param oldF pointer to old float array
* @param oldI pointer to old int array
* @param N number of floats/ints to copy
*/
__global__ void moveEm(float * newF, int *newI, float *oldF, int *oldI, int N) {
  int i=blockIdx.x * blockDim.x + threadIdx.x;
  if(i<N){
    newF[i]=oldF[i];
    newI[i]=oldI[i];
  }
}


/**
* @brief performs a sum of each successive pair of N numbers in source and stores the sums in sums. intended to be run multiple times to sum over a whole array. if N is odd, the last sum index will be N/2-1 and contain the sum of the last 3 numbers
*
* @param sums where to store the sums
* @param source where to get the numbers to sum together
* @param N the dimension of source
*
* @return 
*/
__global__ void sumEm(float *sums, float *source, int N){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=(i<<1);
  if(j+3<N)sums[i]=source[j]+source[j+1];
  else if(j+3==N) sums[i]=source[j]+source[j+1]+source[j+2];
  else if(j+2==N) sums[i]=source[j]+source[j+1];
}
/**
* @brief performs a sum of pairs of N numbers in source, using locations indicated by pointers. pointers has indices multiplied by genomeSize. intended to be run multiple times to sum over a whole array. if N is odd, the last sum index will be N/2-1 and contain the sum of the last 3 numbers
*
* @param sums where to store the sums
* @param source where to get the numbers to sum together
* @param N the dimension of source
* @param ptrs the indices to use when gathering pairs for summation
* @param genomeSize the number by which the indices in ptrs are scaled
*
* @return 
*/
__global__ void sumEmIndex(float *sums, float *source, int N, const int *ptrs, const int genomeSize){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=(i<<1);
  if(j+3<N)sums[i]=source[ptrs[j]/genomeSize]+source[ptrs[j+1]/genomeSize];
  else if(j+3==N) sums[i]=source[ptrs[j]/genomeSize]+source[ptrs[j+1]/genomeSize]+source[ptrs[j+2]/genomeSize];
  else if(j+2==N) sums[i]=source[ptrs[j]/genomeSize]+source[ptrs[j+1]/genomeSize];
#if DEBUG>1
  if(j+2<=N)printf(" %d:%f",i,sums[i]);
#endif
}

/**
* @brief get sum of all areas
*
* @param areas_d pointer to areas on device
* @param ptrs_d pointer to indices for each individual in population
* @param pSize population size
* @param temp_d pointer to temporary array on device
* @param genomeSize number of alleles in genome
*/
float *getSumAreas(float *areas_d, int *ptrs_d, int pSize, float *temp_d, const int & genomeSize){
  int dim=pSize;
  int offset=0;
  sumEmIndex <<<((dim>>1)+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (temp_d, areas_d, dim, ptrs_d, genomeSize);
#if DEBUG>1
  std::cout << std::endl;
#endif
  pSize >>= 1;
  while((dim>>=1)>1){
    offset^=pSize;
    sumEm <<<((dim>>1)+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (temp_d+offset, temp_d+(offset^pSize), dim);
#if DEBUG>1
  std::cout << std::endl;
#endif
  }
  return temp_d+offset;
}

int main(int argc, char *argv[]){
  cudaError_t error;
  std::cout << "Get pSize" << std::endl;
  int pSize=atoi(argv[1]);
  std::cout << "Get nGen" << std::endl;
  int nGen=atoi(argv[2]);
  std::cout << "Get pMut" << std::endl;
  float pMut=atof(argv[3]);
  std::cout << "Get max" << std::endl;
  float max=atof(argv[4]);
  std::cout << "Get pCross" << std::endl;
  float pCross=atof(argv[5]);
  std::cout << "Get rseed" << std::endl;
  int rseed=atoi(argv[6]);
  float *rands;
  /* GPU arrays */
  float *rands_d;
  int genomeSize;
  size_t nRands;
  curandGenerator_t gen;
  int g;
  float *Vs, *Vs_d;
  int *ptrs_d, *ptrs;
  float *tset_d, *tgts_d, *wts_d, *tset, *tgts, *wts, *xx_d, *scores_d, *scores, *areas_d, sumAreas;
  int N, nConf=0, save=pSize/10, *breaks, *breaks_d, nBreaks;
  std::string saveFile,loadFile;
  for (int i=7;i<argc;i++){
    if(i+1<argc){
      if(argv[i][0]=='-'&&argv[i][1]=='r')saveFile=argv[++i];
      else if(argv[i][0]=='-'&&argv[i][1]=='c')loadFile=argv[++i];
    }
  }
  /* load things */
  std::map<std::string,DihCorrection> correctionMap;
  std::cout << "LOAD" << std::endl;
  load(std::cin, &tset, &tgts, &wts, &nConf, &breaks, &nBreaks, &genomeSize, correctionMap);
#if DEBUG && 0
  for(int i=0;i<nConf;i++){
    for(int j=0;j<genomeSize;j++)
      std::cerr << ' ' << tset[i*genomeSize+j];
    std::cerr << std::endl;
  }
  std::cerr << tgts[0] << ' ' << tgts[1] << ' ' << tgts[2] << ' ' << tgts[3] << std::endl;
  std::cerr << "first cudaMalloc, " << nBreaks << " breaks" << std::endl;
#endif
  cudaMalloc((void **)&breaks_d, nBreaks*sizeof(int));
  cudaMalloc((void **)&tgts_d, (nBreaks-1+nConf*(1+genomeSize))*sizeof(float));
  wts_d=tgts_d+nConf;
  tset_d=wts_d+nBreaks-1;
#if DEBUG
  std::cerr << "COPY" << std::endl;
#endif
  cudaMemcpy(breaks_d, breaks, nBreaks*sizeof(breaks[0]), cudaMemcpyHostToDevice);
  if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
  cudaMemcpy(tset_d, tset, nConf*genomeSize*sizeof(float), cudaMemcpyHostToDevice);
  if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
  cudaMemcpy(tgts_d, tgts, nConf*sizeof(float), cudaMemcpyHostToDevice);
  if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
  cudaMemcpy(wts_d, wts, (nBreaks-1)*sizeof(*wts), cudaMemcpyHostToDevice);
  if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}

  /* we need randoms, new pop 3xcrossover, genomeSizexmut */
  nRands=(3+genomeSize)*pSize;
  int nBlocks=(pSize+BLOCK_SIZE-1)/BLOCK_SIZE;
#ifdef DEBUG
  std::cerr << nBlocks << " blocks\n";
#endif
#if DEBUG
  printf("Allocate memory\n");
#endif
  rands=(float *)malloc(nRands*sizeof(float));
  //cudaMalloc((void **)&rands_d, nRands*sizeof(float));
  N=(pSize<<1);
  HANDLE_ERROR(cudaMalloc((void **)&Vs_d, (N*(genomeSize+4)+pSize*nConf+nRands)*sizeof(float)));
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
  rands_d=Vs_d+N*genomeSize;
  scores_d=rands_d+nRands;
  areas_d=scores_d+(N<<1);
  xx_d=areas_d+(N<<1);
  scores=(float *)malloc(sizeof(*scores)*N);
  float *scores_ds[2];
  scores_ds[0]=scores_d;
  scores_ds[1]=scores_d+N;


  Vs=(float *)malloc(N*genomeSize*sizeof(float));

  ptrs=(int *)malloc(sizeof(int)*N);
  ptrs[0]=0;
  for(g=1;g<N;g++)ptrs[g]=ptrs[g-1]+genomeSize;
  HANDLE_ERROR(cudaMalloc((void **)&ptrs_d, N*2*sizeof(int)));
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
  int *ptrs_ds[2];
  ptrs_ds[0]=ptrs_d;
  ptrs_ds[1]=ptrs_d+N;
  cudaMemcpy(ptrs_d, ptrs, sizeof(int)*N, cudaMemcpyHostToDevice);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
  int curList=0;

#if 0
  HANDLE_ERROR(cudaMalloc((void **)&scores_d, N*sizeof(float)));

  HANDLE_ERROR(cudaMalloc((void **)&xx_d, nOffspring*nConf*sizeof(float)));
#endif

  thrust::device_ptr<int> dPtrs(ptrs_d), dPtrs_save(ptrs_d+save);
  thrust::device_ptr<float> dScores(scores_d), dVs(Vs_d);
  thrust::device_ptr<float> dScores_save(scores_d+save),
                            dScores_pSize(scores_d+pSize),
                            dScores_N(scores_d+N);

  //thrust::sequence(dPtrs, dPtrs+N, 0, genomeSize);
  //thrust::fill_n(dVs, pSize*genomeSize, 0.0f);
  //thrust::fill_n(dScores, N, 0.0f);

#if DEBUG
  printf("Create random generator\n");
#endif
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
#if DEBUG
  printf("Seed random generator\n");
#endif
  curandSetPseudoRandomGeneratorSeed(gen, rseed);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (seed)\n", cudaGetErrorString(error));}
#if DEBUG
   std::cerr << "GenerateNormal" << std::endl;
#endif
    curandGenerateNormal(gen, Vs_d, N*genomeSize, 0, 1);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (normal)\n", cudaGetErrorString(error));}
  if(!loadFile.empty()) {
    std::ifstream loadS(loadFile.c_str(), std::ios::in | std::ios::binary);
    loadS.read((char*)Vs,pSize*genomeSize*sizeof(*Vs));
    cudaMemcpy(Vs_d, Vs, pSize*genomeSize*sizeof(*Vs), cudaMemcpyHostToDevice);
  }
  

#if DEBUG
    std::cerr << "1stscore" << std::endl;
#endif
    scoreIt <<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (scores_ds[curList], areas_d, Vs_d, ptrs_ds[curList], tset_d, tgts_d, wts_d, breaks_d, nConf, pSize, genomeSize, xx_d);
    scoreIt <<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (scores_ds[curList]+pSize, areas_d, Vs_d, ptrs_ds[curList]+pSize, tset_d, tgts_d, wts_d, breaks_d, nConf, pSize, genomeSize, xx_d);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (1stscore)\n", cudaGetErrorString(error));}
#if DEBUG
    std::cerr << "1stsort" << std::endl;
#endif
thrust::sort_by_key(thrust::device_pointer_cast(scores_ds[curList]), thrust::device_pointer_cast(scores_ds[curList]+N), thrust::device_pointer_cast(ptrs_ds[curList]));
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (1stsort)\n", cudaGetErrorString(error));}
#if DEBUG>2
    cudaMemcpy(scores, scores_ds[curList], sizeof(*scores)*N, cudaMemcpyDeviceToHost);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
    cudaMemcpy(Vs, Vs_d, sizeof(*Vs)*N*genomeSize, cudaMemcpyDeviceToHost);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
    cudaMemcpy(ptrs, ptrs_ds[curList], sizeof(*ptrs)*N, cudaMemcpyDeviceToHost);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
    for(int i=0;i<N;i++){
      std::cerr << i << ": [" << ptrs[i] << "] = " << scores[i] << " {"<<Vs[ptrs[i]]<<" "<<Vs[ptrs[i]+1]<<" "<<Vs[ptrs[i]+2]<<" "<<Vs[ptrs[i]+3]<<"}\n";
    }
#endif

  for(g=0;g<nGen;g++){
#if DEBUG>1
  printf("Generate random numbers\n");
    printf(" %d",g);fflush(stdout);
#endif
    curandGenerateUniform(gen, rands_d, nRands);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
    //sumAreas=getSumAreas (areas_d, ptrs_ds[curList], I//
#if DEBUG>2
    std::cerr << "Mate" << std::endl;
#endif
    calcAreas <<<nBlocks, BLOCK_SIZE>>> (scores_ds[curList], areas_d, ptrs_d, pSize, genomeSize);
    mateIt <<<nBlocks, BLOCK_SIZE>>> (Vs_d, ptrs_ds[curList], areas_d, 
  getSumAreas(areas_d, ptrs_ds[curList], pSize, areas_d+N, genomeSize),
 //sumAreas,
//thrust::reduce(thrust::device_pointer_cast(areas_d),thrust::device_pointer_cast(areas_d+pSize), 0,thrust::plus<float>()),
 rands_d, pCross, pSize, genomeSize);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (mate)\n", cudaGetErrorString(error));}
#if DEBUG>2
    std::cerr << "Mutate" << std::endl;
#endif
    mutateIt <<<nBlocks, BLOCK_SIZE>>> (Vs_d, ptrs_ds[curList]+pSize, rands_d+pSize*3, pSize, pMut, max, genomeSize);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (mutate)\n", cudaGetErrorString(error));}
    /* score those beyond pSize */
#if DEBUG>2
    std::cerr << "Score" << std::endl;
#endif
    scoreIt <<<nBlocks, BLOCK_SIZE>>> (scores_ds[curList]+pSize, areas_d, Vs_d, ptrs_ds[curList]+pSize, tset_d, tgts_d, wts_d, breaks_d, nConf, pSize, genomeSize, xx_d);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s (score)\n", cudaGetErrorString(error));}
#if DEBUG>2
    std::cerr << "Display em:\n\tCopy scores" << std::endl;
    cudaMemcpy(scores, scores_ds[curList], sizeof(*scores)*N, cudaMemcpyDeviceToHost);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
    std::cerr << "\tCopy Vs" << std::endl;
    cudaMemcpy(Vs, Vs_d, sizeof(*Vs)*N*genomeSize, cudaMemcpyDeviceToHost);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
    std::cerr << "\tCopy ptrs" << std::endl;
    cudaMemcpy(ptrs, ptrs_ds[curList], sizeof(*ptrs)*N, cudaMemcpyDeviceToHost);
    if((error=cudaGetLastError())!=cudaSuccess){fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(error));}
    for(int i=0;i<N;i++){
      std::cerr << i << ": [" << ptrs[i] << "] = " << scores[i] << " {"<<Vs[ptrs[i]]<<" "<<Vs[ptrs[i]+1]<<" "<<Vs[ptrs[i]+2]<<" "<<Vs[ptrs[i]+3]<<"}\n";
    }
#endif

    /* sort & select */
#if DEBUG>2
    std::cerr << "Move 1" << std::endl;
#endif
    moveEm <<<(save+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (scores_ds[curList^1], ptrs_ds[curList^1], scores_ds[curList], ptrs_ds[curList], save);
#if DEBUG>2
    std::cerr << "Move 2" << std::endl;
#endif
    moveEm <<<(pSize+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (scores_ds[curList^1]+save, ptrs_ds[curList^1]+save, scores_ds[curList]+pSize, ptrs_ds[curList]+pSize, pSize);//nOffspring);
#if DEBUG>2
    std::cerr << "Move 3" << std::endl;
#endif
    moveEm <<<(pSize-save+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>> (scores_ds[curList^1]+save+pSize, ptrs_ds[curList^1]+save+pSize, scores_ds[curList]+save, ptrs_ds[curList]+save, pSize-save);
    curList^=1;

    /* first sort only the ones that aren't going to be saved (elitist) */
#if DEBUG>1
    std::cerr << "Selection sort (" << N << " items, less " << save << ")" << std::endl;
#endif
thrust::sort_by_key(thrust::device_pointer_cast(scores_ds[curList]+save), thrust::device_pointer_cast(scores_ds[curList]+pSize+save), thrust::device_pointer_cast(ptrs_ds[curList]+save));

    /* then sort all those that fit within pSize */
#if DEBUG>1
    std::cerr << "Rank sort" << std::endl;
#endif
    thrust::sort_by_key(thrust::device_pointer_cast(scores_ds[curList]), thrust::device_pointer_cast(scores_ds[curList]+pSize), thrust::device_pointer_cast(ptrs_ds[curList]));
  }

  cudaMemcpy(Vs, Vs_d, sizeof(float)*genomeSize*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(ptrs, ptrs_ds[curList], sizeof(int)*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(scores, scores_ds[curList], sizeof(float)*N, cudaMemcpyDeviceToHost);
  for(int i=0;i<pSize;i++){
  std::cout << std::fixed << scores[i] << std::endl;

  for(std::map<std::string,DihCorrection>::iterator it=correctionMap.begin(); it!=correctionMap.end(); ++it){
    std::cout << it->second.setGenome(Vs+ptrs[i]);
  }
  }
  if(!saveFile.empty()){
    std::ofstream saveS(saveFile.c_str(), std::ios::out | std::ios::binary);
    for(int i=0;i<pSize;i++)
      saveS.write((char *)(Vs+ptrs[i]),genomeSize*sizeof(*Vs));
  }
#if 0
  std::cout << scores[pSize] << std::endl;
  for(std::map<std::string,DihCorrection>::iterator it=correctionMap.begin(); it!=correctionMap.end(); ++it){
    std::cout << it->second.setGenome(Vs+ptrs[pSize]);
    //std::cout << it->second;
  }
#endif
  free(ptrs);
#if 0
  printf("Copy random numbers\n");
  cudaMemcpy(rands, rands_d, nRands*sizeof(unsigned int), cudaMemcpyDeviceToHost);

  printf("Print random numbers\n");
  printf("%d",rands[0]);
  for(i=1;i<nRands;i++){
    printf(" %d",rands[i]);
  }
  putchar('\n');
#endif

  curandDestroyGenerator(gen);
  //cudaFree(xx_d);
  cudaFree(Vs_d);
  cudaFree(ptrs_d);
  cudaFree(breaks_d);
  cudaFree(tgts_d);
  free(Vs);
  free(scores);
  //cudaFree(rands_d);
  free(rands);
  return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <list>
#include <map>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

class DihCorrection {
private:
// maps periodicity to column
std::map<int,int> corrs;
std::string label;
const float *genome;
public:
DihCorrection() {}
DihCorrection(std::string label) : label(label) {

}
void addCorr(int n, int col) {
  corrs[n]=col;
}
std::map<int,int> getCorrs() { return corrs; }
const DihCorrection & setGenome(const float *genome) {
  this->genome=genome;
  return *this;
}
friend std::ostream & operator<<(std::ostream &os, const DihCorrection &dc);
};
std::ostream & operator<<(std::ostream &os, const DihCorrection &dc) {
  for(std::map<int,int>::const_iterator it=dc.corrs.begin(), next=it;it!=dc.corrs.end();it=next){
    ++next;
    os << dc.label << std::fixed;
    if(dc.label.length()<=11)os << std::setw(8) << 1;
    float val=dc.genome[it->second];
    os << std::setw(12) << (val<0?-val:val) << std::setw(8) << (val<0?180:0) << std::setw(5) << (next==dc.corrs.end()?it->first:-it->first) << std::endl;
  }
  return os;
}
void load(std::istream & in, float **tset, float **tgts, float **wts, int *nConfs, int **brks, int *nBrks, int *genomeSize, std::map<std::string,DihCorrection> & correctionMap)
{
  int ch;
  int col=0;
  while(in.peek()=='+'||in.peek()=='-'){
    std::string label;
    std::string dih;
    ch=in.get();
    int strLen=(ch=='-'?11:35);
    in >> label;
    in >> std::ws;
    for(int i=0;i<strLen;i++){
      dih.push_back(in.get());
    }
#if LOAD_DEBUG
    std::cout << "Dihedral[" << label  << "]=" << dih << std::endl;
#endif
    DihCorrection dc(dih);
    while((ch=in.get())==' ');
    if(ch=='\n'){
      for(int n=4;n>0;n--) {
#if LOAD_DEBUG
        std::cout << n << ": " << col << std::endl;
#endif
        dc.addCorr(n, col++);
      }
#if LOAD_DEBUG
      std::cout << dc.getCorrs().size() << std::endl;
#endif
    } else {
      while(ch!='\n'){
        if(ch<'0'||ch>'9'){
          std::string otherLabel;
          do {
            otherLabel.push_back(ch);
            ch=in.get();
          } while(ch<'0'||ch>'9');
          dc.addCorr(ch-'0',correctionMap[otherLabel].getCorrs()[ch-'0']);
        } else {
          dc.addCorr(ch-'0', col++);
        }
        do ch=in.get();while(ch==' ');
      }
    }
    correctionMap[label]=dc;
    //std::map<std::string,DihCorrection>::iterator it=correctionMap.find(label);
    //if(it!=)
  }
  *genomeSize=col;
  std::vector<std::vector<float> > data;
  std::vector<int> breaks;
  std::vector<float> weights;
  *nConfs=0;
  std::string line;
  double off;
  while(in.good()&&std::getline(in,line)){
    breaks.push_back(*nConfs);
    std::list<DihCorrection*> cols;
    std::string label;
    std::istringstream input(line);
    input >> label;
#if LOAD_DEBUG
    std::cout << "Residue=" << label << std::endl;
#endif
    weights.push_back(1.0f);
    if(input.good()){
     input >> label;
     if(label[0]=='<') weights.back()=atof(label.data()+1);
     else { cols.push_back(&correctionMap[label]); }
     while(input.good()){
      input >> label;
      cols.push_back(&correctionMap[label]);
     }
    }
    std::vector<float> dataRow;
    while(std::getline(in,line)&&line[0]!='/'){
      input.clear();
      input.str(line);
      dataRow.assign(1+*genomeSize, 0);
      double dih;
      for(std::list<DihCorrection*>::iterator it=cols.begin();it!=cols.end();++it){
        input >> dih;
#if LOAD_DEBUG
        std::cout << dih << ":";
#endif
        dih*=3.141592653589793238/180.;
#if LOAD_DEBUG
        std::cout << (*it)->getCorrs().size() << std::endl;
#endif
#if 0
        for(std::map<int,int>::iterator jt=(*it)->getCorrs().begin();jt!=(*it)->getCorrs().end();++jt){
        //for(const auto& jt:(*it)->getCorrs()){
#if LOAD_DEBUG
          std::cout << " " << jt->first << "[" << jt->second << "]+=" << cos(dih*(float)jt->first);
#endif
          dataRow[jt->second]+=cos(dih*(float)jt->first);
        }
#endif
        for(int n=4;n>0;--n){
#if LOAD_DEBUG
          std::cout << " " << n << "[" << (*it)->getCorrs()[n] << "]+=" << cos(dih*(float)n);
#endif
          dataRow[(*it)->getCorrs()[n]]+=cos(dih*(double)n);
        }
#if LOAD_DEBUG
        std::cout << ' ' << (*it)->getCorrs().size() << std::endl;
#endif
      }
      double E,E0;
      input >> E >> E0;
      if(*nConfs==breaks.back()){
        off=E0-E;
        E=0;
      }else{
        E=E-E0+off;
      }
      dataRow[*genomeSize]=(float)E;
#if LOAD_DEBUG
    std::cout << " deltaE="<<dataRow[*genomeSize]<<std::endl;
#endif
      ++*nConfs;
      data.push_back(dataRow);
    }
    weights.back()/=(float)((*nConfs-breaks.back())*(*nConfs-breaks.back()-1)/2);
  }
  breaks.push_back(*nConfs);
  *nBrks=breaks.size();
#if LOAD_DEBUG
  std::cout << *nConfs << " confs and " << *nBrks << " breaks" << std::endl;
#endif
  *brks=(int *)malloc(sizeof(**brks)*breaks.size());
  *wts=(float *)malloc(sizeof(**wts)*weights.size());
  *tset=(float *)malloc(sizeof(**tset)* *nConfs* *genomeSize);
  *tgts=(float *)malloc(sizeof(**tgts)* *nConfs);
  for(int i=0;i<*nConfs;i++){
    (*tgts)[i]=data[i][*genomeSize];
    for(int j=0;j<*genomeSize;j++){
      (*tset)[i* *genomeSize+j]=data[i][j];
    }
  }
  for(int i=0;i<weights.size();i++){
    (*brks)[i]=breaks[i];
    (*wts)[i]=weights[i];
  }
  for(int i=weights.size();i<*nBrks;i++){
    (*brks)[i]=breaks[i];
  }
}


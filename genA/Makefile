BIN				:= genA
CXXFLAGS		:= -O3 -g


CUDA_INSTALL_PATH ?= $$(which nvcc | sed -e 's@/bin/nvcc@@')

NVCC ?= nvcc
LIBS = -L$(CUDA_INSTALL_PATH)/lib -L$(CUDA_INSTALL_PATH)/lib/x86_64-linux-gnu -lcuda -lcudart -lcublas -lcurand
NVCCFLAGS := 

CPP_SOURCES		:=
CU_SOURCES		:= genA.cu
HEADERS			:= $(wildcard *.h)
CPP_OBJS		:= $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS			:= $(patsubst %.cu, %.cu_o, $(CU_SOURCES))

%.cu_o : %.cu
	$(NVCC) $(NVCCFLAGS) -c $(INCD) -o $@ $<

$(BIN) : $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) $(INCD) $(LIBS)

clean:
	rm -f $(BIN) *.o *.cu_o

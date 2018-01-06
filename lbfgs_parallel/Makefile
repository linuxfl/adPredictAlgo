config = ./config.mk

include $(config)
include ./dmlc.mk

BIN = lbfgs.dmlc

# common build script for programs

export LDFLAGS= -L../../lib -pthread -lm -lrt -lglog $(DMLC_LDFLAGS)
export CFLAGS = -Wall  -msse2  -Wno-unknown-pragmas -fPIC -I./include/rabit/include  -I./include/dmlc-core/include -std=c++11 -fopenmp

CFLAGS+= -I./deps/include
LDFLAGS+= -L./deps/lib

lbfgs.dmlc : lbfgs.cc ./lib/libdmlc.a ./lib/librabit.a

all: $(BIN) 

$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a,  $^) $(LDFLAGS)

clean:
	$(RM) $(OBJ) $(BIN)  *~ ../src/*~


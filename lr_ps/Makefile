CXX = g++

DEPS_PATH = ../common

LDFLAGS = -lpthread -lrt
CFLAGS = -O3 -Wall -Iinclude -std=c++11

CFLAGS += -I$(DEPS_PATH)/include -I./ps-lite/deps/include -I./ps-lite/include/ps -I./ps-lite/include

DIR_SRC = ./src

SRC = $(wildcard $(DIR_SRC)/*.cc)

OBJ = $(patsubst %.cc,$(DIR_SRC)/%.o,$(notdir $(SRC)))
OBJ += ./ps-lite/deps/lib/libprotobuf.a ./ps-lite/deps/lib/libzmq.a ./ps-lite/build/libps.a

target = ./bin/lr_ps

$(target) : $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

$(DIR_SRC)/%.o : $(DIR_SRC)/%.cc
	$(CXX) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

.PHONY:clean

clean:
	$(RM) $(DIR_SRC)/*.o
	$(RM) ./bin/*

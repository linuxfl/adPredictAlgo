CXX = g++

DEPS_PATH = ../common

LDFLAGS = -lpthread
CFLAGS = -O3 -Wall -Iinclude -std=c++0x

CFLAGS += -I$(DEPS_PATH)/include

DIR_SRC = ./src

SRC = $(wildcard $(DIR_SRC)/*.cc)

OBJ = $(patsubst %.cc,$(DIR_SRC)/%.o,$(notdir $(SRC)))
OBJ += $(DEPS_PATH)/lib/libdmlc.a

target = ./bin/adpa_ftrl

$(target) : $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

$(DIR_SRC)/%.o : $(DIR_SRC)/%.cc
	$(CXX) $(CFLAGS) -c $< -o $@

.PHONY:clean

clean:
	$(RM) $(DIR_SRC)/*.o
	$(RM) ./bin/adpa_ftrl

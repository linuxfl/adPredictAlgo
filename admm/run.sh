#!/bin/bash

mpirun -np 2 ./bin/admm ./conf/lbfgs.conf  train_data=./data/shitu_train_

#!/bin/bash

mpirun -np 2 ./bin/admm ./conf/ftrl.conf  train_data=shitu_train_

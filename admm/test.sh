#!/bin/bash

mpirun -np 1 ./bin/admm conf/ftrl.conf task=pred num_fea=10000007 train_data=shitu_test_ model_in=lr_model.dat

#!/bin/bash
if [[ $# -lt 1 ]]
then
    echo "Usage: nprocess"
    exit -1
fi

rm -rf *.model
k=$1

# run linear model, the program will automatically split the inputs
dmlc-core/tracker/dmlc-submit -n $k lbfgs.dmlc shitu_train reg_L2=0.1 

#./lbfgs.dmlc ../data/agaricus.txt.test task=pred model_in=final.model

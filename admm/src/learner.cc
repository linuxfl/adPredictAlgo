#include <iostream>
#include <cstring>

#include "lbfgs.h"
#include "ftrl.h"
#include "adagrad.h"
#include "sparse_lbfgs.h"

namespace adPredictAlgo {

Learner* Learner::Create(const char *name) {
    if(!strcmp(name,"lbfgs")) {
        return new LBFGSSolver();
    }else if(!strcmp(name,"ftrl")) {
        return new FTRL();
    }else if(!strcmp(name,"adagrad")) {
        return new AdaGrad();
    }else if(!strcmp(name,"sparse_lbfgs")) {
        return new SparseLBFGSSolver();
    }


    return nullptr;
}

}

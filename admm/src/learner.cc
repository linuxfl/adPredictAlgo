#include <iostream>
#include <cstring>

#include "lbfgs.h"
#include "ftrl.h"
#include "sgd.h"

namespace adPredictAlgo {

Learner* Learner::Create(const char *name) {
    if(!strcmp(name,"lbfgs")) {
        return new LBFGSSolver();
    }else if(!strcmp(name,"ftrl")) {
        return new FTRL();
    }else if(!strcmp(name,"sgd")) {
        return new SGD();
    }

    return nullptr;
}

}

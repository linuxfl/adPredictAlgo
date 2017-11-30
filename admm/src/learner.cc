#include <iostream>
#include <cstring>

#include "lbfgs.h"
#include "ftrl.h"

namespace adPredictAlgo {

Learner* Learner::Create(const char *name) {
    if(!strcmp(name,"lbfgs")) {
        return new LBFGSSolver();
    }else if(!strcmp(name,"ftrl")) {
        return new FTRL();
    }

    return nullptr;
}

}

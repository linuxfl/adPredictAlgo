#include <iostream>
#include <cstring>

#include "lbfgs.h"
#include "ftrl.h"

namespace adPredictAlgo {

static Learner* Learner::Create(char *name) {
    if(!strcmp(name,"lbfgs")) {
        return new LBFGSSolver();
    }else if(!strcmp(name,"ftrl")) {
        return new FTRL();
    }else{
        return nullptr;
    }
}

}

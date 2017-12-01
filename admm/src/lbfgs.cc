#ifndef _ADPREDICTALGO_LBFGS_H_
#define _ADPREDICTALGO_LBFGS_H_

#include "learner.h"

class LBFGSSolver : public Learner {
    public:
        LBFGSSolver() {
        }
        ~LBFGSSolver() {
        }

        void Configure(std::vector<std::pari<std::string,std::string> > cfg);
        void Train(float *primal,float *dual,float *cons,
                    float rho,dmlc::RowBlockIter<unsigned> *dtrain);
}

#endif

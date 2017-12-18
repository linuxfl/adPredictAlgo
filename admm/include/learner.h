#ifndef _ADPREDICTALGO_LEARN_H_
#define _ADPREDICTALGO_LEARN_H_

#include <iostream>
#include <dmlc/data.h>
#include <vector>

namespace adPredictAlgo {

class Learner {
  public:
    virtual ~Learner(){}
    // configure
    virtual void Configure(const std::vector<std::pair<std::string,std::string> > &) = 0;

    //train
    virtual void Train(float * primal,
                       float *dual,
                       float *cons,
                       float rho,dmlc::RowBlockIter<unsigned> *dtrain) = 0;
        //pred instance
    virtual float PredIns(const dmlc::Row<unsigned> &v,
                          const float *w) = 0;

    static Learner *Create(const char *name);
};

}

#endif

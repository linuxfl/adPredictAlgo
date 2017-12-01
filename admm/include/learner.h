#ifndef _ADPREDICTALGO_LEARN_H_
#define _ADPREDICTALGO_LEARN_H_

#include <iostream>

class Learner {
    public:
        virtual ~Learner();
        // configure
        virtual void Configure(
                std::vector<std::pair<std::string,std::string> >) = 0;

        //prama: 
        virtual void Train(float * primal,float *dual,
                           float *cons,float rho,dmlc::RowBlockIter<unsigned> *dtrain) = 0;

        static Learner *Create(char *name);
};

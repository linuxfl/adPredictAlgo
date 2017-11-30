#ifndef _ADPREDICT_ADMM_H_
#define _ADPREDICT_ADMM_H_

#include <iostream>
#include "dmlc/data.h"
#include "learner.h"

namespace adPredictAlgo {

class ADMM {
    public:
        ADMM(std::vector<std::pair<std::string,std:string> > cfg,
             dmlc::RowBlockIter<unsigned> *dtrain);
        //init
        void Init();
        //update primal parameter x
        void UpdatePrimal();
        //update dual parameter y
        void UpdateDual();
        //update z
        void UpdateConsensus();
        //soft threshold
        void SoftThreshold();
        //train task
        void TaskTrain();
        //predict task
        void TaskPred();
        //save model
        void SaveModel();
        //dump model
        void DumpModel();
        bool IsStop();

    private:
        float *primal,*dual,*cons;
        float *w,*cons_pre;

        uint32_t num_fea;
        uint32_t num_procs;
        int max_iter;

        float l1_reg;
        float l2_reg;

        //optimator
        Learner *leaner;
        LogisticReg lr_model;
};// class ADMM

}// namespace adPredictAlgo

#endif

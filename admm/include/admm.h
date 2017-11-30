#ifndef _ADPREDICT_ADMM_H_
#define _ADPREDICT_ADMM_H_

#include <iostream>
#include "dmlc/data.h"
#include "learner.h"

namespace adPredictAlgo {

struct ADMMParam : public dmlc::Parameter<ADMMParam>
{
    //number of feature
    uint32_t num_fea;
    //the admm paramter
    float rho;
    //the max iter of admm
    int admm_max_iter;

    float l1_reg;
    float l2_reg;

    std::string learner;

    ADMMParam() {
        memset(this,0,sizeof(ADMMParam));
    }

    DMLC_DECLARE_PARAMETER(ADMMParam) {
        DMLC_DECLARE_FIELD(num_fea)
            .set_default(0)
            .describe("the number of the feature.");
        DMLC_DECLARE_FIELD(rho)
            .set_default(1.0);
        DMLC_DECLARE_FIELD(admm_max_iter)
            .set_default(10)
            .set_range(0,100);
        DMLC_DECLARE_FIELD(l1_reg)
            .set_default(0.0f);
        DMLC_DECLARE_FILED(l2_reg)
            .set_default(0.0f);
        DMLC_DECLARE_FILED(learner)
            .set_default("NULL");
    }
};

DMLC_REGISTER_PARAMETER(ADMMParam);

class ADMM {
  public:
    ADMM()
        :primal(nullptr),dual(nullptr)
        ,cons(nullptr),w(nullptr),cons_pre(nullptr)
    {

    }

    virtual ~ADMM() {
        if(primal != nullptr)
            delete [] primal;
        if(dual != nullptr)
            delete [] dual;
        if(cons != nullptr)
            delete [] cons;
        if(cons_pre != nullptr)
            delete [] cons_pre;
        if(w != nullptr)
            delete [] w;
    }

    //init
    inline void Init() {
        CHECK(param.num_fea != 0 || leaner != "NULL") << "num_fea and name_leaner must be set!";
        //init paramter 
        primal = new float[param.num_fea];
        dual = new float[param.num_fea];
        cons = new float[param.num_fea];
        cons_pre = new float[param.num_fea];
        w = new float[param.num_fea];

        //learner
        learner = Learner::Create(param.learner);
        if(learner == nullptr)
            LOG(FATAL) << "learner inital error!";
    }
    
    void Configure(
        std::vector<std::pair<std::string,std::string> > cfg)
    {
        param.InitAllowUnknow(cfg);
        for(const &kv : cfg)
            cfg_[kv.first] = kv.second;
    }

    void UpdatePrimal()
    {

    }

    //update dual parameter y
    void UpdateDual() {

    }
    //update z
    void UpdateConsensus() {

    }
    //soft threshold
    void SoftThreshold() {

    }

    //train task
    void TaskTrain() {

    }

    //predict task
    void TaskPred() {

    }

    //save model
    void SaveModel() {

    }
    //dump model
    void DumpModel() {

    }

    bool IsStop() {

    }

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
    
    ADMMParam param;
    std::map<std::string,std::string> cfg_;
};// class ADMM

}// namespace adPredictAlgo

#endif

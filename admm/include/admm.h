#ifndef _ADPREDICT_ADMM_H_
#define _ADPREDICT_ADMM_H_

#include <iostream>
#include "dmlc/data.h"
#include "dmlc/io.h"
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
    int num_procs;

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
        DMLC_DECLARE_FIELD(l2_reg)
            .set_default(0.0f);
        DMLC_DECLARE_FIELD(learner)
            .set_default("NULL");
        DMLC_DECLARE_FIELD(num_procs)
            .set_default(0);
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
        CHECK(param.num_fea != 0 || param.learner != "NULL") << "num_fea and name_leaner must be set!";
        //init paramter 
        primal = new float[param.num_fea];
        dual = new float[param.num_fea];
        cons = new float[param.num_fea];
        cons_pre = new float[param.num_fea];
        w = new float[param.num_fea];
    }
    
    void Configure(
        std::vector<std::pair<std::string,std::string> > cfg)
    {
        param.InitAllowUnknown(cfg);
        for(const auto &kv : cfg)
            cfg_[kv.first] = kv.second;
        
        //learner
        learner = Learner::Create(param.learner.c_str());
        if(learner == nullptr)
            LOG(FATAL) << "learner inital error!";
        learner->Configure(cfg);
    }

    void UpdatePrimal()
    {
        learner->Train(primal,dual,cons,rho,dtrain);
    }

    //update dual parameter y
    void UpdateDual() {
        for(uint32_t i = 0;i < num_fea;++i) {
            dual[i] += rho * (dual[i] - cons[i]);
        }
    }
    //update z
    void UpdateConsensus() {
        float s = 1.0/(rho * num_procs + 2 * l2_reg);
        float t = s * l1_reg;

        for(uint32_t i = 0;i < num_fea;i++)
        {
            w[i] = s * (primal[i] + dual[i]);
            cons_pre[i] = cons[i];
        }
    
        MPI_Allreduce(w, cons,  num_fea, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    
        if(l1_reg != 0.0f)
            SoftThreshold(t,cons);
    }

    //soft threshold
    void SoftThreshold(float t_,float *z) {
        for(uint32_t i = 0;i < num_fea;i++){
            if(z[i] > t_){
                z[i] -= t_;
            }else if(z[i] <= t_ && z[i] >= -t_){
                z[i] = 0.0;
            }else{
                z[i] += t_;
            }
        }
    }

    //train task
    void TaskTrain() {
        int iter = 0;
        while(++iter < admm_max_iter) {
            this->UpdatePrimal();
            this->UpdateDual();
            this->UpdateConsensus();
            if(IsStop())
                break;
        }
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
			return false;
    }

  private:
    dmlc::RowBlockIter<unsigned> *dtrain;
    float *primal,*dual,*cons;
    float *w,*cons_pre;

    uint32_t num_fea;
    uint32_t num_procs;
    int admm_max_iter;

		float rho;

    float l1_reg;
    float l2_reg;

    //optimator
    Learner *learner;
    
    ADMMParam param;
    std::map<std::string,std::string> cfg_;
};// class ADMM

}// namespace adPredictAlgo

#endif

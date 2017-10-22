#ifndef _ADPREDICTALGO_LBFGS_H_
#define _ADPREDICTALGO_LBFGS_H_

#include <dmlc/data.h>
#include "linear.h"

namespace adPredictAlgo{

class LBFGSSolver{

  public: 
    LBFGSSolver(dmlc::RowBlockIter<unsigned> *dtrain)
      :dtrain(dtrain)
    {
      l1_reg = 0.0f;
      linesearch_c1 = 0.1f;
      linesearch_backoff = 0.5f;
      lbfgs_stop_tol = 1e-4f;
      memory_size = 4;
      
      max_lbfgs_iter = 100;
      max_linesearch_iter = 5;

    }
    
    inline void Init() {
      linear.Init();
      size_t num_fea = std::max(linear.num_fea,dtrain->NumCol());
        
      CHECK(num_fea > 0) << "please init num fea!";
      
      grad = Eigen::VectorXf::Zero(num_fea);
      z = Eigen::VectorXf::Zero(num_fea);      
        alpha.resize(memory_size,0.0f);
      for(size_t i = 0 ;i < memory_size;i++) {
        Eigen::VectorXf yelem(num_fea);
        Eigen::VectorXf selem(num_fea);
        
        y.push_back(yelem);
        s.push_back(selem);
      }

      //init grad
      linear.CalGrad(grad,linear.old_weight,dtrain);
      //init obj val
      init_objval = linear.Eval(dtrain,linear.old_weight);
      old_objval = init_objval;
      LOG(INFO) << "L-BFGS solver starts, num_fea=" << num_fea << ",init_objval=" << init_objval
                << ",memory_size=" << memory_size << ",l1_reg="<<l1_reg << ",l2_reg=" << linear.l2_reg;
    }
    
    inline void SetParam(const char *name,const char *val) {
      if(!strcmp(name,"l1_reg")) l1_reg = static_cast<float>(atof(val));
      if(!strcmp(name,"linesearch_c1")) linesearch_c1 = static_cast<float>(atof(val));
      if(!strcmp(name,"linesearch_backoff")) linesearch_backoff = static_cast<float>(atof(val));
      if(!strcmp(name,"max_linesearch_iter")) max_linesearch_iter = static_cast<int>(atoi(val));
      if(!strcmp(name,"lbfgs_stop_tol")) lbfgs_stop_tol = static_cast<float>(atof(val));
      if(!strcmp(name,"max_lbfgs_iter")) max_lbfgs_iter = static_cast<float>(atof(val));
      if(!strcmp(name,"memory_size")) memory_size = static_cast<int>(atoi(val));

      linear.SetParam(name,val);
    }

    virtual void FindChangeDirection(int iter) {
      int k = iter;
      int M = memory_size;
      int j;
    
        z = grad;
      k - M >= 0? j = M - 1:j = k - 1;
      for(int i = j; i >= 0;i--){
        alpha[i] = s[i].dot(z)/y[i].dot(s[i]);
        z.noalias() -=  alpha[i] * y[i];
      }
      //init H0
      if(k != 0){
        int pre_k = (k - 1)  % M;
        z = s[pre_k].dot(y[pre_k])/y[pre_k].dot(y[pre_k]) * z;
      }
      for(int i = 0;i <= j;i++){
        z.noalias() += s[i] * (alpha[i] - y[i].dot(z)/y[i].dot(s[i]));
      }      
    }

    virtual int BacktrackLineSearch(Eigen::VectorXf &new_weight,
                                    Eigen::VectorXf &old_weight) 
    {
      int k = 0;
      float alpha_ = 1.0;
      float backoff = linesearch_backoff;
      float c1 = linesearch_c1;
      float dginit = 0.0,dgtest;
      
      dginit = grad.dot(-z);
      if(dginit > 0){
        LOG(FATAL) << "The s point is not decent direction." ;
        //return alpha;
      }
      dgtest = c1 * dginit;

      while(k < max_linesearch_iter){
        new_weight = old_weight - alpha_ * z; 
        new_objval = linear.Eval(dtrain,new_weight);
        if(new_objval <= old_objval +  alpha_ * dgtest)
          break;
        else
          alpha_ *= backoff;
        k++;
      }
      return k;
    }

    virtual void UpdateHistInfo(int iter) {
        int k = iter;
        y[k % memory_size] = grad;
        linear.CalGrad(grad,linear.new_weight,dtrain);
        y[k % memory_size] = grad - y[k % memory_size];
        s[k % memory_size] = linear.new_weight - linear.old_weight;
        linear.old_weight = linear.new_weight;
    }



    virtual bool UpdateOneIter(int iter) {
      bool stop = false;
      FindChangeDirection(iter);
      int k = BacktrackLineSearch(linear.new_weight,linear.old_weight);
      UpdateHistInfo(iter);
      if(old_objval - new_objval < lbfgs_stop_tol * init_objval) 
        return true;  
        LOG(INFO) << "[" << iter <<"]" << " L-BFGS: linesearch finishes in "<< k 
                    << " rounds, new_objval=" << new_objval << ", improvment=" << old_objval - new_objval;
        old_objval = new_objval;
      return stop;
    }

    virtual void Run() {
        this->Init();
      int iter = 0;
      while(iter < max_lbfgs_iter) {
        if(this->UpdateOneIter(iter)) break;
        iter++;
      }
    }

  private:
    float l1_reg;
    float linesearch_c1;
    float linesearch_backoff;
    int max_linesearch_iter;
    float lbfgs_stop_tol;
    int max_lbfgs_iter;
    size_t memory_size;

    LinearModel linear;
  
    //obj
    double new_objval;
    double old_objval;
    double init_objval;

    //parameter
    std::vector<float> alpha;
    std::vector<Eigen::VectorXf> y;
    std::vector<Eigen::VectorXf> s;
    
    Eigen::VectorXf z;
    Eigen::VectorXf grad;
    
    //data
    dmlc::RowBlockIter<unsigned> *dtrain;
};

}

#endif

#ifndef _ADPREDICTALGO_LEARN_H_
#define _ADPREDICTALGO_LEARN_H_

#include <iostream>
#include <Eigen/Dense>
#include <dmlc/data.h>
#include <cstring>

//Logistic Regression Model
namespace adPredictAlgo {

struct LogisticReg {
    
  size_t num_fea;
  Eigen::VectorXf old_weight;
  Eigen::VectorXf new_weight;
  float l2_reg;
  
  LinearModel(void){
    num_fea = 0;
    l2_reg = 0.0f;
  }

  inline void Init(size_t _num_fea) {
      num_fea = _num_fea;
    old_weight = Eigen::VectorXf::Zero(num_fea);
    new_weight = Eigen::VectorXf::Zero(num_fea);
  }
  inline void SetParam(const char *name,const char *val) {
    if(!strcmp(name,"l2_reg")) {
      l2_reg = static_cast<float>(atof(val));
    }
    if(!strcmp(name,"num_fea")) {
      num_fea = static_cast<size_t>(atoi(val));
    }
  }
  
  inline double Sigmoid(float inx) const {  
    return 1.0 / (1 + std::exp(-inx));
  }

  
  inline double CalLoss(float label,float margin) const {
    double nlogprob = 0.0;
    if(int(label) == 1) {
      nlogprob = std::log(Sigmoid(margin));
    }else {
      nlogprob = std::log(1-Sigmoid(margin));
    }
    return -nlogprob;
  }

  inline double InnerProduct(const Eigen::VectorXf &w,
                              const dmlc::Row<unsigned> &v) const {
    double sum = 0.0;
    for(unsigned i = 0;i < v.length;i++) {
      if(v.index[i] < num_fea) {
        sum += w[v.index[i]];
      }
    }
    return sum;
  }

  inline double Pred(const Eigen::VectorXf &w,
                            const dmlc::Row<unsigned> &v) const {
    return Sigmoid(InnerProduct(w,v));
  }

  inline double PredToGrad(const Eigen::VectorXf &w,
                            const dmlc::Row<unsigned> &v) const {
    return Sigmoid(InnerProduct(w,v)) - v.get_label();
  }

  virtual void CalGrad(Eigen::VectorXf &out_grad,const Eigen::VectorXf &weight,
        dmlc::RowBlockIter<unsigned> *dtrain) const {
    std::vector<double> grad;
    out_grad.setZero();
    dtrain->BeforeFirst();
    while(dtrain->Next()) {
      const dmlc::RowBlock<unsigned> &batch = dtrain->Value();
      grad.resize(batch.size,0.0f);
      for(size_t i = 0;i < batch.size;i++) {
        dmlc::Row<unsigned> v = batch[i];
        grad[i] = PredToGrad(weight,v);
      }
      for(size_t i = 0;i < batch.size;i++) {
        dmlc::Row<unsigned> v = batch[i];
        for(size_t j = 0; j < v.length;j++) {
          out_grad[v.index[j]] += grad[i];
        }
      }
    } 
    if(l2_reg != 0.0f) {
      for(size_t i = 0;i < num_fea;i++) {
        out_grad[i] += l2_reg * weight[i];
      }
    }
  }
  
  virtual double Eval(dmlc::RowBlockIter<unsigned> *dtrain,
                  const Eigen::VectorXf &weight) {
    double sum_val = 0.0f;
    dtrain->BeforeFirst();
    while(dtrain->Next()) {
      const dmlc::RowBlock<unsigned> &batch = dtrain->Value();
      for(size_t i = 0;i < batch.size;i++) {
        double fv 
            = CalLoss(batch[i].get_label(),InnerProduct(weight,batch[i]));
        sum_val += fv;
      }
    }

    if(l2_reg != 0.0f) {
      double sum_sqr = 0.5 * l2_reg * weight.norm();
      sum_val += sum_sqr;
    }
    return sum_val;
  }

};

}
#endif

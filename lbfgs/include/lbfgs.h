#ifndef _ADPREDICTALGO_LBFGS_H_
#define _ADPREDICTALGO_LBFGS_H_

#include <dmlc/data.h>
#include <fstream>
#include "linear.h"
#include "metric.h"

namespace adPredictAlgo{

// <pscore , label>
typedef struct pair{
  pair(float s,int l):score(s),t_label(l) { } 
  float score;
  int t_label;
  
  bool operator < (const struct pair &p) const {
    return score > p.score;
  }
}pair_t;

class LBFGSSolver{

  public:
    std::string task;
    std::string model_out;
    std::string model_in;

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
      max_linesearch_iter = 20;
      num_fea = 0;

        task = "train";
        model_out = "lr_model.dat";
        model_in = "NULL";
    }
    
    virtual ~LBFGSSolver() {
      if(dtrain != nullptr)
        delete dtrain;
    }

    inline void Init() {
      num_fea = std::max(num_fea,dtrain->NumCol());
      linear.Init(num_fea);
 
      CHECK(num_fea > 0) << "please init num fea!";
      
      grad = Eigen::VectorXf::Zero(num_fea);
      l1_grad = Eigen::VectorXf::Zero(num_fea);

      z = Eigen::VectorXf::Zero(num_fea);
      alpha.resize(memory_size,0.0f);
      for(size_t i = 0 ;i < memory_size;i++) {
        Eigen::VectorXf yelem(num_fea);
        Eigen::VectorXf selem(num_fea);
        
        y.push_back(yelem);
        s.push_back(selem);
      }

      //init gradient
      linear.CalGrad(grad,linear.old_weight,dtrain);
      //init l1 gradient;
      if(l1_reg != 0) SetDirL1Sign(l1_grad,grad,linear.old_weight);
      else l1_grad = grad;
      //init obj val
      init_objval = this->Eval(dtrain,linear.old_weight);
      old_objval = init_objval;
      LOG(INFO) << "L-BFGS solver starts, num_fea=" << num_fea << ",init_objval=" << init_objval
                << ",memory_size=" << memory_size << ",l1_reg="<<l1_reg << ",l2_reg=" << linear.l2_reg;
    }
    
    inline void SetParam(const char *name,const char *val) {
      if(!strcmp(name,"l1_reg")) 
          l1_reg = static_cast<float>(atof(val));
      if(!strcmp(name,"linesearch_c1")) 
          linesearch_c1 = static_cast<float>(atof(val));
      if(!strcmp(name,"linesearch_backoff")) 
          linesearch_backoff = static_cast<float>(atof(val));
      if(!strcmp(name,"max_linesearch_iter")) 
          max_linesearch_iter = static_cast<int>(atoi(val));
      if(!strcmp(name,"lbfgs_stop_tol")) 
          lbfgs_stop_tol = static_cast<float>(atof(val));
      if(!strcmp(name,"max_lbfgs_iter")) 
          max_lbfgs_iter = static_cast<float>(atof(val));
      if(!strcmp(name,"memory_size")) 
          memory_size = static_cast<int>(atoi(val));
      if(!strcmp(name,"num_fea"))
          num_fea = static_cast<size_t>(atoi(val));
      if(!strcmp(name,"task"))
          task = val;
      if(!strcmp(name,"model_in"))
          model_in = val;
      if(!strcmp(name,"model_out"))
          model_out = val;
      
      linear.SetParam(name,val);
    }

    //return l1_reg_grad and direction dot
    virtual float FindChangeDirection(int iter) {
      int k = iter;
      int M = memory_size;
      int j;
    
      z = l1_grad;
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
      //SetDirL1Sign(l1_grad,grad,linear.old_weight);
      FixL1Sign(z,l1_grad);
      return l1_grad.dot(-z);
    }

    virtual void FixL1Sign(Eigen::VectorXf &p,Eigen::VectorXf &l1_grad) {
      if(l1_reg != 0.0f){
        for(size_t i = 0; i < num_fea;i++) {
          if(p[i] * l1_grad[i] <= 0.0f) {
            p[i] = 0.0f;
          }
        }
      }
    }

    virtual void SetDirL1Sign(Eigen::VectorXf &out_dir,
                    const Eigen::VectorXf &grad,const Eigen::VectorXf &weight) {
      if(l1_reg == 0.0f){
            return;
      }
      
      for(size_t i = 0;i < num_fea;i++) {
        if(weight[i] == 0.0f){
          if(grad[i] > l1_reg){
            out_dir[i] = grad[i] - l1_reg;
          }else if(grad[i] < -l1_reg) {
            out_dir[i] = grad[i] + l1_reg;
          }else {
            out_dir[i] = 0.0f;
          }
        }else{
          if(weight[i] > 0.0f){
            out_dir[i] = grad[i] + l1_reg;
          }else{
            out_dir[i] = grad[i] - l1_reg;
          }
        }
      }
    }
    
    virtual void FixWeightSign(Eigen::VectorXf &new_weight,
                            const Eigen::VectorXf &old_weight,const Eigen::VectorXf &l1_grad) {
      if(l1_reg != 0.0f) {
        for(size_t i = 0;i < num_fea;i++) {
          if(old_weight[i] == 0.0f){
            if(-l1_grad[i] * new_weight[i] < 0.0f) {
              new_weight[i] = 0.0f;
            }
          }else if(old_weight[i] * new_weight[i] < 0.0f){
             new_weight[i] = 0.0f;
          }
        }
      }
    }

    virtual int BacktrackLineSearch(Eigen::VectorXf &new_weight,
                                const Eigen::VectorXf &old_weight,float dot_dir_l1grad,int iter) 
    {
      int k = 0;
      float alpha_ = 1.0;
      float backoff = linesearch_backoff;
      float c1 = linesearch_c1;
      float dginit = 0.0,dgtest;
      
      if(iter == 0) {
          alpha_ = 1.0f / std::sqrt(-dot_dir_l1grad);
          backoff = 0.1f;
        }
      //direction dot gradient
      //dginit = grad.dot(-z);
      if(dginit > 0){
        LOG(FATAL) << "The s point is not decent direction." ;
        //return alpha;
      }
      dgtest = c1 * dot_dir_l1grad;

      while(k < max_linesearch_iter){
        new_weight = old_weight - alpha_ * z;
        FixWeightSign(new_weight,old_weight,l1_grad);
        new_objval = this->Eval(dtrain,new_weight);
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
        if(l1_reg == 0.0f)
            l1_grad = grad;
        else
            SetDirL1Sign(l1_grad,grad,linear.new_weight);
        y[k % memory_size] = grad - y[k % memory_size];
        s[k % memory_size] = linear.new_weight - linear.old_weight;
        linear.old_weight = linear.new_weight;
    }

    virtual bool UpdateOneIter(int iter) {
      bool stop = false;
      float vdot = FindChangeDirection(iter);
      int k = BacktrackLineSearch(linear.new_weight,linear.old_weight,vdot,iter);
      UpdateHistInfo(iter);
      if(old_objval - new_objval < lbfgs_stop_tol * init_objval) 
        return true;  
      LOG(INFO) << "[" << iter <<"]" << " L-BFGS: linesearch finishes in "<< k 
                    << " rounds, new_objval=" << new_objval << ", improvment=" << old_objval - new_objval;
      old_objval = new_objval;
      return stop;
    }

    virtual void TaskPred(void) {
      num_fea = std::max(num_fea,dtrain->NumCol());
      Eigen::VectorXf weight = Eigen::VectorXf::Zero(num_fea);
      
      std::ifstream is(model_in.c_str());
      CHECK(is.fail() == false) << "open model file error!";
      
      for(size_t i = 0;i < num_fea;i++)
        is >> weight[i];
    
      std::vector<Metric::pair_t> pair_vec;
      dtrain->BeforeFirst();
      while(dtrain->Next()) {
        const dmlc::RowBlock<unsigned> &batch = dtrain->Value();
        for(size_t i = 0;i < batch.size;i++) {
          dmlc::Row<unsigned> v = batch[i];
          for(unsigned j = 0; j < v.length;j++) {
            double pv = linear.Pred(weight,v);
            Metric::pair_t p(pv,v.get_label());
            pair_vec.push_back(p);
          }
        }
      }
        
      is.close();
      
      LOG(INFO) << "test AUC is : " << Metric::CalAUC(pair_vec) 
                << ", test COPC is : " << Metric::CalCOPC(pair_vec);
    }
    
    virtual void SaveModel() {
      std::ofstream os(model_out.c_str());
      
      CHECK(os.fail() == false) << "open model file fail";
      for(size_t i = 0;i < num_fea;i++) {
        os << linear.old_weight[i] << std::endl;
      }
      os.close();
    }

    virtual void TaskTrain(void) {
      this->Init();
      int iter = 0;
      while(iter < max_lbfgs_iter) {
        if(this->UpdateOneIter(iter)) break;
        iter++;
      }
      this->SaveModel();
    }

    virtual void Run(void) {
      if(task == "train") {
        TaskTrain();
      }else if(task == "pred") {
        TaskPred();
      }else{
        LOG(FATAL) << "unspecfied task!!!";
      }
    }

    virtual double Eval(dmlc::RowBlockIter<unsigned> *dtrain,
                            const Eigen::VectorXf &weight) {
      double fv = linear.Eval(dtrain,weight);
        
      if (l1_reg) {
        for(size_t i = 0;i < num_fea;i++) {
          fv += l1_reg * std::abs(weight[i]);
        }
      }
      return fv;
    }

/*   double CalAUC(std::vector<pair_t> p) {
     long total_score = 0;
     long pos_num = 0;

     sort(p.begin(),p.end());
     size_t num = p.size();

     for(size_t i = 0;i < num;i++) {
       if(p[i].t_label == 1) {
         total_score = total_score + num - i;
         pos_num++;
       }   
     }   
     total_score = total_score - pos_num * (pos_num + 1)/2.0f;
     return (total_score * 1.0f / ((num - pos_num) * pos_num));
    }    
    
    double CalCOPC(const std::vector<pair_t> &p) {
      float score = 0.0;
      int label = 0;
      for(auto iter = p.begin();iter != p.end();iter++) {
        score += (*iter).score;
        label += (*iter).t_label;
      }   
      return label * 1.0 / score;
    }   
*/
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
    size_t num_fea;

    //parameter
    std::vector<float> alpha;
    std::vector<Eigen::VectorXf> y;
    std::vector<Eigen::VectorXf> s;
    
    Eigen::VectorXf z;
    Eigen::VectorXf grad;
    Eigen::VectorXf l1_grad;

    //data
    dmlc::RowBlockIter<unsigned> *dtrain;
};

}

#endif

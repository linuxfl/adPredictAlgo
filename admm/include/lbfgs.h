#include "learner.h"
#include "Eigen/Dense"
#include <omp.h>

namespace adPredictAlgo {

class LBFGSSolver : public Learner {
  public:
    LBFGSSolver() {
      linesearch_c1 = 1e-4f;
      linesearch_backoff = 0.5f;
      max_linesearch_iter = 20;
      lbfgs_stop_tol = 1e-5f;
      max_lbfgs_iter = 10;
      memory_size = 4;
      num_fea = 0;
      nthread = 0;
      debug = false;

      Eigen::VectorXf e(num_fea);
      y.resize(memory_size,e);
      s.resize(memory_size,e);

      alpha.resize(memory_size,0.0);

      z = Eigen::VectorXf::Zero(num_fea);
      grad = Eigen::VectorXf::Zero(num_fea);
      new_w = Eigen::VectorXf::Zero(num_fea);
    }

    ~LBFGSSolver() {
      y.clear();
      s.clear();
      alpha.clear();
    }

    void Configure(const std::vector<std::pair<std::string,std::string> > &cfg) 
    {
      for(const auto kv : cfg)
        cfg_[kv.first] = kv.second;

      if(cfg_.count("linesearch_c1"))
        linesearch_c1 = static_cast<float>(atof(cfg_["linesearch_c1"].c_str()));
      if(cfg_.count("linesearch_backoff"))
        linesearch_backoff = static_cast<float>(atof(cfg_["linesearch_backoff"].c_str()));
      if(cfg_.count("max_linesearch_iter"))
        max_linesearch_iter = static_cast<int>(atoi(cfg_["max_linesearch_iter"].c_str()));
      if(cfg_.count("lbfgs_stop_tol"))
        lbfgs_stop_tol = static_cast<float>(atof(cfg_["lbfgs_stop_tol"].c_str()));
      if(cfg_.count("max_lbfgs_iter"))
        max_lbfgs_iter = static_cast<float>(atof(cfg_["max_lbfgs_iter"].c_str()));
      if(cfg_.count("memory_size"))
        memory_size = static_cast<int>(atoi(cfg_["memory_size"].c_str()));
      if(cfg_.count("num_fea"))
        num_fea = static_cast<size_t>(atoi(cfg_["num_fea"].c_str()));
      if(cfg_.count("nthread"))
        nthread = static_cast<int>(atoi(cfg_["nthread"].c_str()));
      if(cfg_.count("debug"))
        debug = static_cast<bool>(atoi(cfg_["debug"].c_str()));
      if(cfg_.count("rank"))
        rank = static_cast<int>(atoi(cfg_["rank"].c_str()));
      this->MemInit();

      if(rank == 0)
        LOG(INFO) << "L-BFGS solver starts, num_fea=" << num_fea << ",memory_size=" << memory_size
                  << ",lbfgs_stop_tol=" << lbfgs_stop_tol << ",max_lbfgs_iter=" << max_lbfgs_iter
                  << ",max_linesearch_iter=" << max_linesearch_iter << ",nthread=" << nthread << ",debug=" << debug;
    }
  
    float PredIns(const dmlc::Row<unsigned> &v,
                  const float *w){
       float inner = 0.0;
       for(unsigned int i = 0; i < v.length;++i)
       {
         if(v.index[i] > num_fea)
            continue;
         inner += w[v.index[i]] * v.get_value(i);
       }
       return Sigmoid(inner);
    }
	
    void Train(float *primal,
               float *dual,
               float *cons,
               float rho,
               dmlc::RowBlockIter<unsigned> *dtrain)
    {
      //bind with the Eigen vector
      Eigen::VectorXf old_w = Eigen::VectorXf::Map(primal, num_fea);
      //Eigen::Map<Eigen::VectorXf> old_w(primal, num_fea);
      Eigen::Map<Eigen::VectorXf> d(dual, num_fea);
      Eigen::Map<Eigen::VectorXf> c(cons, num_fea);

      this->ParamInit(old_w, d, c, rho, dtrain);
      int iter = 0;
      while(iter < max_lbfgs_iter) {
        if(this->UpdateOneIter(old_w, d, c, rho, iter, dtrain))
          break;
        iter++;
      }
      //primal = new_w.data();
      for(size_t i = 0;i < num_fea;i++){
        primal[i] = new_w[i];
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
    size_t num_fea;
    int nthread;

    bool debug;
    int rank;

    double old_objval;
    double new_objval;
    double init_objval;

    std::vector<float> alpha;
    std::vector<Eigen::VectorXf> y;
    std::vector<Eigen::VectorXf> s;

    Eigen::VectorXf z;
    Eigen::VectorXf grad;
    Eigen::VectorXf new_w;
    std::map<std::string,std::string> cfg_;

  private:
    inline void MemInit()
    {
      Eigen::VectorXf e(num_fea);
      y.resize(memory_size,e);
      s.resize(memory_size,e);

      alpha.resize(memory_size,0.0);

      z = Eigen::VectorXf::Zero(num_fea);
      grad = Eigen::VectorXf::Zero(num_fea);
      new_w = Eigen::VectorXf::Zero(num_fea);
    }

    inline void ParamInit(const Eigen::VectorXf & old_w,
                     const Eigen::VectorXf & dual,
                     const Eigen::VectorXf & cons,
                     const float &rho,
                     dmlc::RowBlockIter<unsigned> *dtrain) {
      //init gradient
      CalGrad(grad, old_w, dual, cons, rho, dtrain);
      //init obj val
      init_objval = this->Eval(old_w, dual, cons, rho, dtrain);
      old_objval = init_objval;
    }

    //one update
    bool UpdateOneIter(Eigen::VectorXf &w,
                       const Eigen::VectorXf &d,
                       const Eigen::VectorXf &c,
                       float rho,
                       int iter,
                       dmlc::RowBlockIter<unsigned> *dtrain)
    {
      bool stop = false;
      float vdot = FindChangeDirection(iter);
      int k = BacktrackLineSearch(w, d, c, rho, vdot, iter, dtrain);
      UpdateHistInfo(w, d, c, rho, iter, dtrain);
      if(old_objval - new_objval < lbfgs_stop_tol * init_objval)
        return true;
      if(rank == 0 && debug)
        LOG(INFO) << "[" << iter <<"]" << " L-BFGS: linesearch finishes in "<< k+1 << " rounds, new_objval="
                      << new_objval << ", improvment=" << old_objval - new_objval;
      old_objval = new_objval;
      return stop;
    }

    float FindChangeDirection(int iter)
    {
      int k = iter;
      int M = memory_size;
      int j,bound,end = 0;

      if(k != 0)
        end = (k-1) % M;
      z = grad;
      //k - M >= 0? j = M - 1:j = k - 1;
      k > M? bound = M - 1:bound = k - 1;
      j = end;
      for(int i = 0; i <= bound;++i) {
        alpha[j] = s[j].dot(z)/y[j].dot(s[j]);
        z.noalias() -=  alpha[j] * y[j];
        j = (j - 1 + M) % M;
      }
      //init H0
      if(k != 0){
        int pre_k = (k - 1)  % M;
        z = s[pre_k].dot(y[pre_k])/y[pre_k].dot(y[pre_k]) * z;
      }
      for(int i = 0;i <= bound;++i){
        j = (j + 1) % M;
        z.noalias() += s[j] * (alpha[j] - y[j].dot(z)/y[j].dot(s[j]));
      }
      //SetDirL1Sign(l1_grad,grad,linear.old_weight);
      return grad.dot(-z);
    }

    int BacktrackLineSearch(Eigen::VectorXf &old_w,
                            const Eigen::VectorXf &d,
                            const Eigen::VectorXf &c,
                            float rho,
                            float dot_dir_grad, int iter,
                            dmlc::RowBlockIter<unsigned> *dtrain)
    {
      int k = 0;
      float alpha_ = 1.0;
      float backoff = linesearch_backoff;
      float c1 = linesearch_c1;
      float dginit = 0.0,dgtest;

      if(iter == 0) {
          alpha_ = 1.0f / std::sqrt(-dot_dir_grad);
          backoff = 0.1f;
        }
      //direction dot gradient
      //dginit = grad.dot(-z);
      if(dginit > 0){
        LOG(FATAL) << "The s point is not decent direction." ;
        //return alpha;
      }
      dgtest = c1 * dot_dir_grad;
      while(k < max_linesearch_iter){
        new_w = old_w - alpha_ * z;
        new_objval = this->Eval(new_w, d, c, rho, dtrain);
        if(new_objval <= old_objval +  alpha_ * dgtest)
          break;
        else
          alpha_ *= backoff;
        k++;
      }
      return k;
    }

    void UpdateHistInfo(Eigen::VectorXf &old_w,
                        const Eigen::VectorXf &d,
                        const Eigen::VectorXf &c,
                        const float & rho,
                        const int &iter,
                        dmlc::RowBlockIter<unsigned> *dtrain)
    {
      int k = iter;
      y[k % memory_size] = grad;
      CalGrad(grad, new_w, d, c, rho, dtrain);
      y[k % memory_size] = grad - y[k % memory_size];
      s[k % memory_size] = new_w - old_w;
      old_w = new_w;
    }

    // grad_i = pred - label + d_i + rho * (old_w_i - c_i)
    void CalGrad(Eigen::VectorXf & grad,
                 const Eigen::VectorXf & old_w,
                 const Eigen::VectorXf & d,
                 const Eigen::VectorXf & c,
                 const float &rho,
                 dmlc::RowBlockIter<unsigned> *dtrain)
    {
      if(nthread != 0) omp_set_num_threads(nthread);
      std::vector<float> grad_;
      //out_grad.setZero();
      grad.setZero();
      dtrain->BeforeFirst();
      while(dtrain->Next()) {
        const dmlc::RowBlock<unsigned> &batch = dtrain->Value();
        grad_.resize(batch.size,0.0f);
        //#pragma omp parallel for schedule(static)
        for(size_t i = 0;i < batch.size;i++) {
          dmlc::Row<unsigned> v = batch[i];
          grad_[i] = PredToGrad(old_w, v);
        }
        //#pragma omp parallel
        for(size_t i = 0;i < batch.size;i++) {
          dmlc::Row<unsigned> v = batch[i];
          for(size_t j = 0; j < v.length;j++) {
            grad[v.index[j]] += grad_[i];
          }
        }
      }
      grad += d + rho * (old_w - c);
    }

    float Eval(const Eigen::VectorXf & old_w,
               const Eigen::VectorXf & d,
               const Eigen::VectorXf & c,
               const float &rho,
               dmlc::RowBlockIter<unsigned> *dtrain)
    {
      if(nthread != 0) omp_set_num_threads(nthread);
      float sum_val = 0.0f;
      dtrain->BeforeFirst();
      while(dtrain->Next()) {
        const dmlc::RowBlock<unsigned> &batch = dtrain->Value();
        //#pragma omp parallel for schedule(static) reduction(+:sum_val)
        for(size_t i = 0;i < batch.size;i++) {
          float fv
                = CalLoss(old_w, batch[i]);
          sum_val += fv;
        }
      }
      float r = d.dot(old_w - c) + rho / 2 * (old_w - c).dot(old_w - c);
      sum_val += r;
      return sum_val;
    }

    inline float Sigmoid(float inx)
    {
      return 1. / (1. + exp(-inx));
    }

	float PredIns(const dmlc::Row<unsigned> &v,
                  const Eigen::VectorXf &old_w){
       float inner = 0.0;
       for(unsigned int i = 0; i < v.length;++i)
       {
         if(v.index[i] > num_fea)
            continue;
         inner += old_w[v.index[i]] * v.get_value(i);
       }
       return Sigmoid(inner);
    }
	
    inline float PredToGrad(const Eigen::VectorXf &old_w,
                            const dmlc::Row<unsigned> &v)
    {
       return PredIns(v, old_w) - v.get_label();
    }

    inline float CalLoss(const Eigen::VectorXf & old_w,
                         const dmlc::Row<unsigned> &v)
    {
       float nlogprob = 0.;
       float pred = PredIns(v, old_w);

       if(int(v.get_label()) == 0)
       {
         nlogprob = std::log(1 - pred);
       }else{
         nlogprob = std::log(pred);
       }
       return -nlogprob;
    }

};

}

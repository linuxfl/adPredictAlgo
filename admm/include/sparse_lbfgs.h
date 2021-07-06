#include "learner.h"
#include "Eigen/Dense"
#include "mpi.h"
#include <set>

namespace adPredictAlgo {

class SparseLBFGSSolver : public Learner {
  public:
    SparseLBFGSSolver() {
      linesearch_c1 = 1e-4f;
      linesearch_backoff = 0.5f;
      max_linesearch_iter = 20;
      lbfgs_stop_tol = 1e-5f;
      max_lbfgs_iter = 10;
      memory_size = 4;
      num_fea = 0;
      filter_threshold = 0; 
      debug = false;
      _is_first_admm_iter = false;
    }

    ~SparseLBFGSSolver() {
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
      if(cfg_.count("debug"))
        debug = static_cast<bool>(atoi(cfg_["debug"].c_str()));
      if(cfg_.count("rank"))
        rank = static_cast<int>(atoi(cfg_["rank"].c_str()));
      if(cfg_.count("filter_threshold"))
        filter_threshold = static_cast<int>(atoi(cfg_["filter_threshold"].c_str()));
 
      this->MemInit();

      if(rank == 0)
        LOG(INFO) << "L-BFGS solver starts, num_fea=" << num_fea << ",memory_size=" << memory_size
                  << ",lbfgs_stop_tol=" << lbfgs_stop_tol << ",max_lbfgs_iter=" << max_lbfgs_iter
                  << ",max_linesearch_iter=" << max_linesearch_iter << ",debug=" << debug;
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

    void FetchLocalUpdateIdx(dmlc::RowBlockIter<unsigned> *dtrain) {
      local_idx_freq.resize(num_fea, 0);
      idx_freq.resize(num_fea, 0);
      dtrain->BeforeFirst();
      while(dtrain->Next()) {
        const dmlc::RowBlock<unsigned> &batch = dtrain->Value();
        for(size_t i = 0;i < batch.size;i++) {
          dmlc::Row<unsigned> v = batch[i];
          for(size_t j = 0; j < v.length;j++) {
            local_idx_freq[v.index[j]] += 1;
          }
        }
      }
      MPI_Allreduce(&local_idx_freq[0], &idx_freq[0], num_fea, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      for(int idx = 0;idx < local_idx_freq.size();idx++) {
        if(local_idx_freq[idx] > 0 && idx_freq[idx] > filter_threshold)
            need_update_idx.push_back(idx);
      }
      if(rank == 0)
        LOG(INFO) << "filter threshold " << filter_threshold << ", need update idx cnt " << need_update_idx.size();
    }

    void Train(float *primal,
               float *dual,
               float *cons,
               float rho,
               dmlc::RowBlockIter<unsigned> *dtrain)
    {
      if (!_is_first_admm_iter) {
        _is_first_admm_iter = true;
        FetchLocalUpdateIdx(dtrain);
      }
      this->ParamInit(primal, dual, cons, rho, dtrain);
      int iter = 0;
      while(iter < max_lbfgs_iter) {
        if(this->UpdateOneIter(primal, dual, cons, rho, iter, dtrain))
          break;
        iter++;
      }
      for(auto idx : need_update_idx){
        primal[idx] = new_w[idx];
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

    bool debug;
    int rank;

    double old_objval;
    double new_objval;
    double init_objval;

    std::vector<float> alpha;
    std::vector<std::vector<float>> y;
    std::vector<std::vector<float>> s;
    std::vector<float> z;
    std::vector<float> grad;
    std::vector<float> new_w;
    std::map<std::string,std::string> cfg_;

    bool _is_first_admm_iter;
    int filter_threshold;
    std::vector<int> idx_freq;
    std::vector<int> local_idx_freq;
    std::vector<unsigned> need_update_idx;

  private:
    inline void MemInit()
    {
      y.resize(memory_size, std::vector<float>(num_fea, 0.0f));
      s.resize(memory_size, std::vector<float>(num_fea, 0.0f));
      alpha.resize(memory_size, 0.0f);
      z.resize(num_fea, 0.0f);
      grad.resize(num_fea, 0.0f);
      new_w.resize(num_fea, 0.0f);
    }

    inline void ParamInit(float *primal,
                          float *dual,
                          float *cons,
                          const float rho,
                          dmlc::RowBlockIter<unsigned> *dtrain) {
      //init gradient
      CalGrad(grad, primal, dual, cons, rho, dtrain);
      //init obj val
      init_objval = this->Eval(primal, dual, cons, rho, dtrain);
      old_objval = init_objval;
    }

    //one update
    bool UpdateOneIter(float *w,
                       float *d,
                       float *c,
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

    float SparseVVDot(const std::vector<float> &v1, const std::vector<float> &v2) {
      float out = 0.0f;
      for(auto idx : need_update_idx) {
        out += v1[idx] * v2[idx];
      }
      return out;
    }

    void SparseVFDot(std::vector<float> &v1, float f) {
      for(auto idx : need_update_idx) {  
        v1[idx] = v1[idx] * f;
      }
    }

    void SparseVVFDot(std::vector<float> &v1, std::vector<float> &v2, float f, char t) {
      for(auto idx : need_update_idx) {
        if(t == 'm') {
            v1[idx] -= v2[idx] * f;
        } else {
            v1[idx] += v2[idx] * f;    
        }
      }    
    }

    float FindChangeDirection(int iter)
    {
      int k = iter;
      int M = memory_size;
      int j, bound,end = 0;

      if(k != 0)
        end = (k-1) % M;
      
      for(auto idx : need_update_idx)
        z[idx] = grad[idx];

      //k - M >= 0? j = M - 1:j = k - 1;
      k > M? bound = M - 1:bound = k - 1;
      j = end;
      for(int i = 0; i <= bound;++i) {
        alpha[j] = SparseVVDot(s[j], z) / SparseVVDot(y[j], s[j]);
        SparseVVFDot(z, y[j], alpha[j], 'm');
        j = (j - 1 + M) % M;
      }
      //init H0
      if(k != 0){
        int pre_k = (k - 1)  % M;
        float t = SparseVVDot(s[pre_k], y[pre_k]) / SparseVVDot(y[pre_k], y[pre_k]);
        SparseVFDot(z, t);
      }
      for(int i = 0;i <= bound;++i){
        j = (j + 1) % M;
        float t = alpha[j] - SparseVVDot(y[j], z)/SparseVVDot(y[j], s[j]);
        SparseVVFDot(z, s[j], t, 'a');
      }
      float vdot = 0.f;
      for(auto idx : need_update_idx) {
        vdot += grad[idx] * z[idx] * -1;    
      }
      return vdot;
    }

    int BacktrackLineSearch(float *w,
                            float *d,
                            float *c,
                            float rho,
                            float dot_dir_grad, int iter,
                            dmlc::RowBlockIter<unsigned> *dtrain)
    {
      int k = 0;
      float alpha_ = 1.0f;
      float backoff = linesearch_backoff;
      float c1 = linesearch_c1;
      float dginit = 0.0f, dgtest;

      if(iter == 0) {
          alpha_ = 1.0f / std::sqrt(-dot_dir_grad);
          backoff = 0.1f;
        }
      if(dginit > 0){
        LOG(FATAL) << "The s point is not decent direction." ;
      }
      dgtest = c1 * dot_dir_grad;
      while(k < max_linesearch_iter){
        for(auto idx : need_update_idx) {
            new_w[idx] = w[idx] - alpha_ * z[idx];    
        }
        new_objval = this->Eval(&new_w[0], d, c, rho, dtrain);
        if(new_objval <= old_objval +  alpha_ * dgtest)
          break;
        else
          alpha_ *= backoff;
        k++;
      }
      return k;
    }

    void UpdateHistInfo(float *w,
                        float *d,
                        float *c,
                        const float & rho,
                        const int &iter,
                        dmlc::RowBlockIter<unsigned> *dtrain)
    {
      int k = iter;
      int offset = k % memory_size;
      for(auto idx : need_update_idx)
        y[offset][idx] = grad[idx];
      CalGrad(grad, &new_w[0], d, c, rho, dtrain);
      for(auto idx : need_update_idx) {
        y[offset][idx] = grad[idx] - y[offset][idx];
        s[offset][idx] = new_w[idx] - w[idx];
        w[idx] = new_w[idx];
      }
    }

    // grad_i = pred - label + d_i + rho * (old_w_i - c_i)
    void CalGrad(std::vector<float> &grad,
                 float *primal,
                 float *d,
                 float *c,
                 const float rho,
                 dmlc::RowBlockIter<unsigned> *dtrain)
    {
      std::vector<float> grad_;
      dtrain->BeforeFirst();
      for(auto idx : need_update_idx)
        grad[idx] = 0;
      while(dtrain->Next()) {
        const dmlc::RowBlock<unsigned> &batch = dtrain->Value();
        grad_.resize(batch.size,0.0f);
        for(size_t i = 0;i < batch.size;i++) {
          dmlc::Row<unsigned> v = batch[i];
          grad_[i] = PredToGrad(primal, v);
        }
        for(size_t i = 0;i < batch.size;i++) {
          dmlc::Row<unsigned> v = batch[i];
          for(size_t j = 0; j < v.length;j++) {
            grad[v.index[j]] += grad_[i] * v.get_value(j);
          }
        }
      }
      for(auto idx : need_update_idx) {
        grad[idx] += d[idx] + rho * (primal[idx] - c[idx]);
      }
    }

    float Eval(float *primal,
               float *dual,
               float *cons,
               const float rho,
               dmlc::RowBlockIter<unsigned> *dtrain)
    {
      float sum_val = 0.0f;
      dtrain->BeforeFirst();
      while(dtrain->Next()) {
        const dmlc::RowBlock<unsigned> &batch = dtrain->Value();
        for(size_t i = 0;i < batch.size;i++) {
          float fv
                = CalLoss(primal, batch[i]);
          sum_val += fv;
        }
      }
      float r = 0.f;
      for(auto idx : need_update_idx) {
        r += dual[idx] * (primal[idx] - cons[idx]) + rho / 2 * \
            (primal[idx] - cons[idx]) * (primal[idx] - cons[idx]);
      }
      sum_val += r;
      return sum_val;
    }

    inline float Sigmoid(float inx)
    {
      return 1. / (1. + exp(-inx));
    }

    inline float PredToGrad(float *w,
                            const dmlc::Row<unsigned> &v)
    {
       return PredIns(v, w) - v.get_label();
    }

    inline float CalLoss(float *w,
                         const dmlc::Row<unsigned> &v)
    {
       float nlogprob = 0.;
       float pred = PredIns(v, w);

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

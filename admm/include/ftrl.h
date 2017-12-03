#include <cmath>
#include "learner.h"

namespace adPredictAlgo {

class FTRL : public Learner {
  public:
    FTRL() {
      z = nullptr;
      n = nullptr;

      num_fea = 0;
      alpha = 0.1f;
      beta = 1.0f;
    }

    virtual ~FTRL() {
      if(z)
        delete [] z;
      if(n)
        delete [] n;
    }

    void Configure(const std::vector<std::pair<std::string,std::string> > &cfg) 
    {
      for(const auto &kv : cfg)
        cfg_[kv.first] = kv.second;

      if(cfg_.count("num_fea"))
        num_fea = static_cast<uint32_t>(atoi(cfg_["num_fea"].c_str()));
      if(cfg_.count("alpha"))
        alpha = static_cast<float>(atof(cfg_["alpha"].c_str()));
      if(cfg_.count("beta"))
        beta = static_cast<float>(atof(cfg_["beta"].c_str()));
 
      this->Init();
    }

    void Train(float *primal,
           float *dual,
           float *cons,
           float rho,
           dmlc::RowBlockIter<unsigned> *dtrain)
    {
      dtrain->BeforeFirst();
      while(dtrain->Next())  {
        const  dmlc::RowBlock<unsigned> &batch = dtrain->Value();
        for(size_t i = 0;i < batch.size;i++)
        {
          dmlc::Row<unsigned> v = batch[i];
          for(unsigned j = 0;j < v.length;j++) {
            unsigned fea_index = v.index[j];
            float z_val = z[fea_index];
            float n_val = n[fea_index];

            primal[fea_index] =  -z_val / ((beta + std::sqrt(n_val)) / alpha);
          }

          float grad = PredIns(v,primal) - v.get_label();
          float grad_tmp = grad;

          for(unsigned j = 0;j < v.length;j++) {
            unsigned fea_index = v.index[j];
            float w_val = primal[fea_index];
            float n_val = n[fea_index];

            //rediual iterm
            grad_tmp += dual[fea_index] + rho * (primal[fea_index] - cons[fea_index]);

            float theta = (std::sqrt(n_val + grad_tmp * grad_tmp) \
                - std::sqrt(n_val)) / alpha;
            z[fea_index] += grad_tmp - theta * w_val;
            n[fea_index] += grad_tmp * grad_tmp;
            grad_tmp = grad;
          }
        }
      }
    }

   private:
    inline void Init()
    {
      CHECK(num_fea != 0) << "num_fea must be init.";
      z = new float[num_fea];
      n = new float[num_fea];
    }

    inline float PredIns(const dmlc::Row<unsigned> &v,
                         const float *w) {
      float inner = 0.0f;
      for(unsigned i = 0;i < v.length;i++) {
        inner += w[v.index[i]];
      }
      return Sigmoid(inner);
    }

    inline float Sigmoid(const float &inx) {
      return 1.0f / (1.0f + std::exp(-inx));
    }

    float *z,*n;
    float alpha,beta;
    uint32_t num_fea;

    std::map<std::string,std::string> cfg_;
};

}

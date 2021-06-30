#include <cmath>
#include "learner.h"

namespace adPredictAlgo {

class AdaGrad : public Learner {
  public:
    AdaGrad() {
      num_fea = 0;
      epochs = 1;
      alpha = 0.1;
      beta = 1;
      sqr_grad = NULL;
    }

    virtual ~AdaGrad() {
      if(sqr_grad != nullptr)
        delete [] sqr_grad;
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
      if(cfg_.count("epochs"))
        epochs = static_cast<int>(atoi(cfg_["epochs"].c_str()));

      Init();
    }

    void Train(float *primal,
               float *dual,
               float *cons,
               float rho,
               dmlc::RowBlockIter<unsigned> *dtrain)
    {
      memset(primal,0.f,sizeof(float)*num_fea);
      memset(sqr_grad,0.f,sizeof(float)*num_fea);

      for(int iter = 0; iter < epochs;iter++){
        dtrain->BeforeFirst();
        while(dtrain->Next())  {
          const dmlc::RowBlock<unsigned> &batch = dtrain->Value();
          for(size_t i = 0;i < batch.size;i++)
          {
            dmlc::Row<unsigned> v = batch[i];
            float grad = PredIns(v,primal) - v.get_label();
            float grad_tmp = grad;

            for(unsigned j = 0;j < v.length;j++) {
              unsigned fea_index = v.index[j]; 
              grad_tmp += dual[fea_index] + rho * (primal[fea_index] - cons[fea_index]);
              sqr_grad[fea_index] += grad_tmp * grad_tmp;
              float lr = alpha / sqrt(sqr_grad[fea_index] + beta);
              primal[fea_index] -= lr * grad_tmp;
              grad_tmp = grad;
            }
          }
        }
      }
    }

    float PredIns(const dmlc::Row<unsigned> &v,
                         const float *w) {
      float inner = 0.0f;
      for(unsigned i = 0;i < v.length;i++) {
        inner += w[v.index[i]] * v.get_value(i);
      }
      return Sigmoid(inner);
    }

    inline float Sigmoid(const float &inx) {
      return 1.0f / (1.0f + std::exp(-inx));
    }

    inline void Init()
    {
      sqr_grad = new float[num_fea];
    }

  private:
    float alpha;
    float beta;
    float *sqr_grad;
    uint32_t num_fea;
    int epochs;

    std::map<std::string,std::string> cfg_;
};

}

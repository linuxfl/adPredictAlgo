#include <cmath>
#include "learner.h"

namespace adPredictAlgo {

class SGD : public Learner {
  public:
    SGD() {
      num_fea = 0;
      epochs = 1;
      alpha = 0.1;
    }

    virtual ~SGD() {
    }

    void Configure(const std::vector<std::pair<std::string,std::string> > &cfg) 
    {
      for(const auto &kv : cfg)
        cfg_[kv.first] = kv.second;

      if(cfg_.count("num_fea"))
        num_fea = static_cast<uint32_t>(atoi(cfg_["num_fea"].c_str()));
      if(cfg_.count("alpha"))
        alpha = static_cast<float>(atof(cfg_["alpha"].c_str()));
        std::cout << alpha << std::endl;
      if(cfg_.count("epochs"))
        epochs = static_cast<int>(atoi(cfg_["epochs"].c_str()));
    }

    void Train(float *primal,
               float *dual,
               float *cons,
               float rho,
               dmlc::RowBlockIter<unsigned> *dtrain)
    {
      memset(primal,0.1,sizeof(float)*num_fea);
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
              primal[fea_index] -= alpha * grad_tmp;
//              std::cout << primal[fea_index] << " " << alpha * grad_tmp << std::endl;
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
        inner += w[v.index[i]];
      }
      return Sigmoid(inner);
    }

    inline float Sigmoid(const float &inx) {
      return 1.0f / (1.0f + std::exp(-inx));
    }

  private:
    float alpha;
    uint32_t num_fea;
    int epochs;

    std::map<std::string,std::string> cfg_;
};

}

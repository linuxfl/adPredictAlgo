#include <iostream>
#include "ps.h"
#include <time.h>
#include "data.h"

namespace adPredictAlgo{

typedef struct FTRLEntry{
  FTRLEntry() : w(0),n(0),z(0)
  {

  }
  ValueType w;
  ValueType n;
  ValueType z;
} ftrlentry;

struct KVServerFTRLHandle {
  void operator()(const ps::KVMeta& req_meta, 
                  const ps::KVPairs<float>& req_data, 
                  ps::KVServer<float>* server) {
    size_t n = req_data.keys.size();
    ps::KVPairs<float> res;
    if (req_meta.push) {
      CHECK_EQ(n, req_data.vals.size());
      for(size_t i = 0;i < n;i++)
      {
        ps::Key key = req_data.keys[i];
        FTRLEntry & e = store[key];
        UpdateW(e, req_data.vals[i]);
      }
    } else {
      res.keys = req_data.keys;
      res.vals.resize(n);
      for(size_t i = 0;i < n;i++)
      {
        KeyType key = req_data.keys[i];
        FTRLEntry & e = store[key];
        res.vals[i] = e.w;
      }
    }
    server->Response(req_meta, res);
  }
  
  public:  
    void Configure(const std::vector<std::pair<std::string, std::string> > &cfg){
      //todo
      for(auto kv : cfg)
        cfg_[kv.first] = kv.second;

      if(cfg_.count("l1_reg"))
        l1_reg = static_cast<float>(atof(cfg_["l1_reg"].c_str()));
      if(cfg_.count("l2_reg"))
        l2_reg = static_cast<float>(atof(cfg_["l2_reg"].c_str()));
      if(cfg_.count("alpha"))
        alpha = static_cast<float>(atof(cfg_["alpha"].c_str()));
      if(cfg_.count("beta"))
        beta = static_cast<float>(atof(cfg_["beta"].c_str()));
    }

  private:
    std::unordered_map<KeyType, ftrlentry> store;
    std::unordered_map<std::string, std::string> cfg_;

    //l1 reg
    float l1_reg = 0.1;
    //l2_reg
    float l2_reg = 0;
    float alpha = 0.01;
    float beta = 1;

    void UpdateW(FTRLEntry &e, ValueType g){
      ValueType w = e.w;
      g += l2_reg * e.w;
      
      ValueType cg = e.n;
      ValueType cg_new = cg + g * g;
      e.n = cg_new;
      e.z -= g - (cg_new - cg) / alpha * w;

      ValueType z = e.z;
      if(z <= l1_reg && z > -l1_reg){
        e.w = 0.;
      }else{
        ValueType eta = (beta + cg_new) / alpha;
        e.w = (z > 0 ? z - l1_reg : z + l1_reg) / eta;
      }
    }
};

class S{
 public:
  S(){ }
  
  void StartServer(const std::vector<std::pair<std::string, std::string> > &cfg){
    auto server_ = new ps::KVServer<float>(0);
    auto ftrl_handle_ = new KVServerFTRLHandle();
    ftrl_handle_->Configure(cfg);
    server_->set_request_handle(*ftrl_handle_);
    std::cout << "Start Server Success " << std::endl;
  }

  ~S(){}

  ps::KVServer<ValueType>* server_;
};//end class Server

}

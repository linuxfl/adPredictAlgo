#include <iostream>
#include <vector>
#include <unordered_map>

#include "dataloader.h"
#include "data.h"
#include "ps.h"

namespace adPredictAlgo {
class W{
 public:
  W()
  {
    kv_ = new ps::KVWorker<ValueType>(0);
    minibatch_size = 200;
    epochs = 1;
  }
  ~W() {}

  ValueType Sigmoid(ValueType x){
    if(x < -30) return 1e-6;
    else if(x > 30) return 1.0;
    return 1. / 1. + std::exp(-x);
  }

  void Configure(const std::vector<std::pair<std::string, std::string>> &cfg)
  {
    for(auto & kv : cfg){
      cfg_[kv.first] = kv.second;
    }

    if(cfg_.count("minibatch_size"))
      minibatch_size = static_cast<size_t>(atoi(cfg_["minibatch_size"].c_str()));
    if(cfg_.count("epochs"))
      epochs = static_cast<int>(atoi(cfg_["epochs"].c_str()));
    cfg_["rank"] = std::to_string(rank);
  }

  void Pull(const DataVec &bk,
            std::vector<KeyType> &all_keys,
            std::unordered_map<KeyType, ValueType> &keys_weight)
  {
    keys_weight.clear();
    all_keys.clear();
    auto w = std::make_shared<std::vector<ValueType>>();
    
    for(auto ins : bk)
      for(auto fid : ins.fea_vec)
        all_keys.push_back(fid);

    sort(all_keys.begin(), all_keys.end(), 
                [](const KeyType &a, const KeyType &b){ return a < b;});
    all_keys.erase(unique(all_keys.begin(), all_keys.end()), all_keys.end());
    kv_->Wait(kv_->Pull(all_keys, &(*w)));
    CHECK_EQ(all_keys.size(), (*w).size());
    for(size_t idx = 0;idx < all_keys.size();idx++)
      keys_weight[all_keys[idx]] = (*w)[idx];
  }

  void Push(const std::vector<KeyType> &all_keys,
            const std::unordered_map<KeyType, ValueType> &grad)
  {
    std::vector<ValueType> push_g;
    for(auto key : all_keys)
      push_g.push_back(grad.at(key));
    kv_->Wait(kv_->Push(all_keys, push_g));
  }

  ValueType PredIns(const Instance &ins,
                    const std::unordered_map<KeyType, ValueType> &keys_weight)
  {
    ValueType sum = 0;
    for(auto fid : ins.fea_vec)
      sum += keys_weight.at(fid);
    return Sigmoid(sum);
  }

  void CalGrad(const DataVec &bk,
               const std::vector<KeyType> &all_keys,
               const std::unordered_map<KeyType, ValueType> &keys_weight,
               std::unordered_map<KeyType, ValueType> &grad)
  {
    grad.clear();
    size_t bk_size = bk.size();
    for(auto ins : bk)
    {
      ValueType g = PredIns(ins, keys_weight) - ins.label;
      for(auto fid : ins.fea_vec){
        grad[fid] += g / bk_size;
      }
    }
  }

  void TaskTrain(){
    DataVec bk_;
    std::unordered_map<KeyType, ValueType> keys_weight;
    std::vector<KeyType> all_keys;
    std::unordered_map<KeyType, ValueType> grad;

    for(int epoch = 0; epoch < epochs; ++epoch){
      size_t begin = 0;
      while(1){
        //get a block
        bk_.clear();
        for(size_t i = begin;i < begin + minibatch_size;i++)
        {
          if(i > data.size())
            break;
          bk_.push_back(data[i]);
        }
        begin += minibatch_size;
        if(rank == 0 && begin / minibatch_size % 100 == 0)
          LOG(INFO) << "Complete 100 minibatch!";

        if(begin > data.size())
          break;
        Pull(bk_, all_keys, keys_weight);
        CalGrad(bk_, all_keys, keys_weight, grad);
        Push(all_keys, grad);
      }
      if(rank == 0)
        LOG(INFO) << "iter=" << epoch;
    }
  }

  void Run(const std::vector<std::pair<std::string, std::string>> &cfg){
    rank = ps::MyRank();

    this->Configure(cfg);
    dl_.Configure(cfg_);    
    
    if(rank == 0)
      LOG(INFO) << "minibatch_size=" << minibatch_size << ", epochs=" << epochs;

    dl_.LoadAllDataFromFile(data);
    TaskTrain();
  }

 public:
  int rank;
  size_t minibatch_size;
  int epochs;

  DataLoader dl_;
  std::unordered_map<std::string, std::string> cfg_;
  DataVec data;

  ps::KVWorker<float>* kv_;
};//end class worker

}

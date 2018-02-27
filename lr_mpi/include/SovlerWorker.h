#ifndef _ADPREDICTALGO_SOVLERWORKER_H_
#define _ADPREDICTALGO_SOVLERWORKER_H_

#include <iostream>
#include <mpi.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>

#include "logisticreg.h"

namespace adPredictAlgo {

class SovlerWorker {
  public:
    SovlerWorker()
    {
      num_fea = 0;
      train_size = 0;
      all_train_size = 0;
    }

    virtual ~SovlerWorker()
    {
    }

    inline void Init()
    {
      assert(num_fea != 0);
    }

    inline void SetParam(const char *name,const char *val)
    {
      if(!strcmp(name,"num_fea"))
        num_fea = static_cast<uint32_t>(atoi(val));
      if(!strcmp(name,"minibatch_size"))
        minibatch_size = static_cast<int>(atoi(val));
      if(!strcmp(name,"num_epochs"))
        num_epochs = static_cast<int>(atoi(val));
      if(!strcmp(name,"rank"))
        rank = static_cast<int>(atoi(val));

      dl_.SetParam(name,val);
    }
    void Start()
    {
      //LOG(INFO) << "SovlerWorker " << rank << " Start"<< ", num_fea=" << num_fea 
      //          << ", minibatch_size=" << minibatch_size; 
      dl_.Init();
      data.clear();
      dl_.LoadAllDataFromFile(data);
      
      //reset minibatch size

      int iter = 0;
      int begin = 0;
      int len = data.size();
      while(iter < num_epochs){
        if(begin > len)
          begin = 0;
        UpdateOneIter(begin);
        begin += minibatch_size;
        iter++;
      }
    }

    void UpdateOneIter(const int begin)
    {
      std::vector<uint32_t> send_keys;
      std::unordered_map<uint32_t, float> keys_weight;
      std::unordered_map<uint32_t, float> grad;
      std::vector<Instance> block;

      for(size_t i = begin;i < begin + minibatch_size;++i)
      {
        
        if(i > data.size())
          break;

        Instance ins = data[i];
        for(size_t k = 0;k < ins.fea_vec.size();++k)
          send_keys.push_back(ins.fea_vec[k]);
        block.push_back(ins);
      }
      
      sort(send_keys.begin(),send_keys.end());
      send_keys.erase(unique(send_keys.begin(), send_keys.end()), send_keys.end());

      //send the keys index
      SendKeysToServer(send_keys);
      //recv the weight to update
      RecvWeightFromServer(keys_weight, send_keys);
      //calc the gradient
      CalGrad(keys_weight, grad, block);
      SendGradToServer(grad, send_keys);
    }

    inline void SendKeysToServer(std::vector<uint32_t> &s_keys)
    {
      int keys_n = (int)s_keys.size();
      MPI_Send(&keys_n, 1, MPI_INT, 0, 99, MPI_COMM_WORLD);
      MPI_Send(&s_keys[0], keys_n, MPI_INT, 0, 100, MPI_COMM_WORLD);
    }

    inline void RecvWeightFromServer(std::unordered_map<uint32_t,float> &keys_weight,
                                     const std::vector<uint32_t> &s_keys)
    {
      MPI_Status status;
      keys_weight.clear();
      int key_n = (int)s_keys.size();
      std::vector<float> w(key_n);
      MPI_Recv(&w[0],key_n, MPI_FLOAT, 0, 101, MPI_COMM_WORLD, &status);
      for(size_t i = 0;i < w.size();i++){
        keys_weight[s_keys[i]] = w[i];
      }
    }

    inline void SendGradToServer(std::unordered_map<uint32_t ,float> grad,
                                 std::vector<uint32_t> s_keys)
    {
      std::vector<float> g;
      for(size_t index = 0;index < s_keys.size();++index)
      {
        g.push_back(grad[s_keys[index]]);
      }
      MPI_Send(&g[0], g.size(), MPI_INT, 0, 102, MPI_COMM_WORLD);
    }

    virtual void CalGrad(std::unordered_map<uint32_t,float> &keys_weight,
                         std::unordered_map<uint32_t ,float> &grad,
                         const std::vector<Instance> & block)
    {
      grad.clear();
      size_t len = block.size();
      for(size_t i = 0;i < len;++i)
      {
        Instance ins = block[i];
        float g = PredIns(keys_weight,ins) - ins.label;
        for(size_t j = 0;j < ins.fea_vec.size();++j)
        {
          if(grad.count(ins.fea_vec[j]))
            grad[ins.fea_vec[j]] += g / len;
          else
            grad[ins.fea_vec[j]] = g / len;
        }
      }
    }

    virtual float PredIns(std::unordered_map<uint32_t ,float> &keys_weight,
                          const Instance &ins)
    {
      float inner = 0.0;
      for(size_t i = 0; i < ins.fea_vec.size();++i)
      {
        uint32_t fea_index = ins.fea_vec[i];
        if(!keys_weight.count(fea_index))
        {
          std::cout <<"unordered_map don't hit " << fea_index << " " << std::endl;
        }else{
          inner += keys_weight[fea_index];
        }
      }
      return Sigmoid(inner);
    }

    inline float Sigmoid(float inx)
    {
      return 1.0 / (1.0 + std::exp(-inx));
    }

  private:
    DataLoader dl_;
    DataVec data;
    size_t num_fea;
    unsigned int minibatch_size;
    int num_epochs;
    int train_size;
    int rank;
    int all_train_size;
};
}

#endif

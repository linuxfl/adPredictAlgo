#ifndef __ADPREDICTALGO_FTRL_TRAIN_
#define __ADPREDICTALGO_FTRL_TRAIN_

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <cmath>
#include <cstring>

#include "ffm.h"

namespace adPredictAlgo {
class FTRL {

public:
  FTRL(char *dtrain)
    :dtrain(dtrain),n(nullptr),
    z(nullptr),n_ffm(nullptr),z_ffm(nullptr)
   {
    alpha = 0.1f;
    beta = 1.0f;
    alpha_ffm = 0.1f;
    beta_ffm = 1.0f;

    l1_reg = 0.0f;
    l2_reg = 0.1f;
    l1_ffm_reg = 0.0f;
    l2_ffm_reg = 0.1f;
  }

  virtual ~FTRL() 
  {
    if(z)
      delete [] z;
    if(n)
      delete [] n;
    if(z_ffm)
      delete [] z_ffm;
    if(n_ffm)
      delete [] n_ffm;
  }

  inline void Init()
  {
    ffm.Init();    

    z = new float[ffm.n];
    n = new float[ffm.n];
    
    z_ffm = new float[ffm.ffm_model_size];
    n_ffm = new float[ffm.ffm_model_size];
  }

  inline void SetParam(const char *name,const char *val) {
    if(!strcmp(name,"alpha"))
      alpha = static_cast<float>(atof(val));
    if(!strcmp(name,"beta"))
      beta = static_cast<float>(atof(val));
    if(!strcmp(name,"alpha_ffm"))
      alpha_ffm = static_cast<float>(atof(val));
    if(!strcmp(name,"beta_ffm"))
      beta_ffm = static_cast<float>(atof(val));
    if(!strcmp(name,"l1_reg"))
      l1_reg = static_cast<float>(atof(val));
    if(!strcmp(name,"l2_reg"))
      l2_reg = static_cast<float>(atof(val));
    if(!strcmp(name,"l1_ffm_reg"))
      l1_ffm_reg = static_cast<float>(atof(val));
    if(!strcmp(name,"l2_ffm_reg"))
      l2_ffm_reg = static_cast<float>(atof(val));

    ffm.SetParam(name,val);
  }

  inline void TaskTrain() 
  {
    std::ifstream train_stream(dtrain);
    CHECK(train_stream.fail() == false) << "open the train file error!";

    std::string line;
    Instance ins;
    while(getline(train_stream,line)) {
      ins.clear();
      ParseLine(line,ins);
      UpdateOneIter(ins);
    }
  }

  inline void TaskPred()
  {
    
  }

  inline void StringSplit(const std::string &line,
                          std::string seperator,
                          std::vector<std::string> &result) {
    std::string::size_type first_pos = line.find(seperator);
    std::string::size_type second_pos = 0;
    while(std::string::npos != first_pos) {
      result.push_back(line.substr(first_pos,second_pos - first_pos));

      first_pos = second_pos + seperator.size();
      second_pos = line.find(seperator);
    }
  }

  virtual void ParseLine(const std::string &line,Instance &ins)
  {
    std::vector<std::string> fea_vec;
    StringSplit(line," ",fea_vec);
    ins.label = static_cast<int>(atoi(fea_vec[0].c_str()));

    for(size_t i = 1;i < fea_vec.size();i++) {
      std::vector<std::string> kvs;
      ffm_node _node;
      StringSplit(fea_vec[i],":",kvs);

      _node.field_index = static_cast<uint32_t>(atoi(kvs[0].c_str()));
      _node.fea_index = static_cast<uint32_t>(atoi(kvs[1].c_str()));

      CHECK(_node.field_index <= ffm.m) << "field index must less then the number of field.";
      CHECK(_node.fea_index < ffm.n) << "fea index must less then the number of fea";

      ins.fea_vec.push_back(_node);
    }
  }

  virtual float PredictRaw(const Instance &ins)
  {
    size_t ins_len = ins.fea_vec.size();
    std::vector<ffm_node> fea_vec;
    float sum = 0.0f;
    for(size_t index = 0;index < ins_len;++index)
    {
      uint32_t fea_index = fea_vec[index].fea_index;
      if(std::fabs(z[fea_index]) < l1_reg) {
        ffm.w[fea_index] = 0.0f;
      }else{
        ffm.w[fea_index] = (Sign(z[fea_index]) * l1_reg - z[fea_index]) / \
                      ((beta + std::sqrt(n[fea_index])) / alpha + l2_reg);
      }
      sum += ffm.w[fea_index];
    }

    for(size_t index = 0;index < ins_len;++index)
    {
      uint32_t fea_index = fea_vec[index].fea_index;
      uint32_t field_index = fea_vec[index].field_index;
      for(size_t k = 0;k < ffm.d;++k) {
        uint32_t real_fea_index = 
                  (fea_index - 1) * ffm.m * ffm.d + (field_index - 1) * ffm.d + k;
        if(std::fabs(z_ffm[real_fea_index]) < l1_ffm_reg){
          ffm.v[real_fea_index] = 0.0f;
        }else{
          ffm.v[real_fea_index] = (Sign(z_ffm[real_fea_index]) * l1_ffm_reg - z_ffm[real_fea_index]) / \
                          ((beta_ffm + std::sqrt(n_ffm[real_fea_index])) / alpha_ffm + l2_ffm_reg);
        }
        sum += ffm.v[real_fea_index];
      }
    }
    return sum;
  }
  
  virtual float Predict(float inx) {
    return Sigmoid(inx);
  }

  inline int Sign(float inx)
  {
    return inx > 0?1:0;
  }

  inline float Sigmoid(float inx)
  {
    return 1.0f / (1.0f + std::exp(-inx));
  }

  virtual void AuxUpdate(const Instance &ins,float grad)
  {
    size_t ins_len = ins.fea_vec.size();
    std::vector<ffm_node> fea_vec = ins.fea_vec;

    for(size_t index = 0;index < ins_len;++index)
    {
      uint32_t fea_index = fea_vec[index].fea_index;
      float theta = (std::sqrt(n[fea_index] + grad * grad) - sqrt(n[fea_index])) / alpha;
      z[fea_index] += grad - theta * ffm.w[fea_index];
      n[fea_index] += grad * grad;
    }

    std::map<uint32_t,float> sum_ffm;
    for(size_t i = 0;i < ins_len;++i)
    {
      uint32_t fea_x = fea_vec[i].fea_index;
      uint32_t field_x = fea_vec[i].field_index;
      uint32_t real_fea_x = 
                  (fea_x - 1) * ffm.m * ffm.d + (field_x - 1) * ffm.d;

      for(size_t j = 0; j < ins_len;++j)
      {
        uint32_t fea_y = fea_vec[j].fea_index;
        uint32_t field_y = fea_vec[j].field_index;
        uint32_t real_fea_y = 
                  (fea_y - 1) * ffm.m * ffm.d + (field_y - 1) * ffm.d;
        
        if(i != j) {
          for(size_t k = 0;k < ffm.d;++k) {
            uint32_t real_index = real_fea_x + k;
            if(sum_ffm.find(real_index) != sum_ffm.end())
              sum_ffm[real_index] += ffm.w[real_fea_y + k];
            else
              sum_ffm[real_index] = ffm.w[real_fea_y + k];
          }
        }
      }
    }

    for(size_t index = 0;index < ins_len;++index)
    {
      uint32_t fea_index = fea_vec[index].fea_index;
      uint32_t field_index = fea_vec[index].field_index;
      for(size_t k = 0;k < ffm.d;++k) {
        uint32_t real_fea_index = 
                  (fea_index - 1) * ffm.m * ffm.d + (field_index - 1) * ffm.d + k;
        float g_ffm = grad * sum_ffm[real_fea_index];
        float theta = (std::sqrt(n_ffm[real_fea_index] + g_ffm * g_ffm) - std::sqrt(n_ffm[real_fea_index]));
        z_ffm[real_fea_index] += g_ffm - theta * ffm.w[real_fea_index];
        n_ffm[real_fea_index] += g_ffm * g_ffm;
      }
    }
  }

  virtual void UpdateOneIter(const Instance &ins) 
  {
    float p = Predict(PredictRaw(ins));
    int label = ins.label;
    float grad = p - label;
    AuxUpdate(ins,grad);
  }

  virtual void Run()
  {
    if(task == "train")
      this->TaskTrain();
    else if(task == "pred")
      this->TaskPred();
    else
      LOG(FATAL) << "error task!";
  }

private:
  char *dtrain;

  FFMModel ffm;
  float *n,*z;
  float *n_ffm,*z_ffm;

  float alpha,beta;
  float alpha_ffm,beta_ffm;

  float l1_reg,l2_reg;
  float l1_ffm_reg,l2_ffm_reg;

  std::string task;
}; //end class

} // end namespace
#endif

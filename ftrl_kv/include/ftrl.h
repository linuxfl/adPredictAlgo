#ifndef _FTRL_H_
#define _FTRL_H_

#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <cassert>

#include "str_util.h"
#include "sparse_fea.h"

namespace algo {

typedef float ValueType;

struct ftrlentry{
  ValueType n;
  ValueType z;
  ValueType w;
  ftrlentry(){
    n = 0; z = 0; w = 0;
  }
};

class FTRLSolver {

public:
  FTRLSolver(char *dtrain):train_data(dtrain)
  {
    alpha = 0.01;
    beta = 1.;
    l1_reg = 0;
    l2_reg = 0.1;

    task = "train";
    model_out = "lr_model.dat";
    model_in = "NULL";
    pred_out = "pred.txt";
    is_incre = 0;
    save_aux = 1;
    only_weight = 0;
  }

  virtual ~FTRLSolver(){ model.clear(); }

  virtual void SetParam(const char *name, const char *val);
  //task train
  virtual void TaskTrain();
  virtual void TrainIns(const fea::instance &ins);
  //task predict
  virtual void TaskPred();
  virtual void SaveModel() const;
  virtual void LoadModel();

  inline void Run(){
    std::cout << "FTRLSovler Start," << " alpha=" << alpha << ", beta=" << beta << ", l1_reg=" << l1_reg << ", only_weight="
              << only_weight << ", l2_reg=" << l2_reg << ", is_incre=" << is_incre << ", save_aux=" << save_aux << std::endl;

    if(task == "train"){
      //if incre model, load model first
      if(is_incre)
        LoadModel();
        
      TaskTrain();
      SaveModel();
    }

    if(task == "pred") {
      LoadModel();
      TaskPred();
    }
  }

  inline int Sign(ValueType inx){ return inx > 0?1:-1;}
  inline ValueType Sigmoid(ValueType inx) {
    return 1./(1 + std::exp(-inx)); 
  }


private:
  using ModelDict = std::unordered_map<uint32_t, ftrlentry>;
  ModelDict model; //lr model
  
  float alpha;
  float beta;
  float l1_reg;
  float l2_reg;

  std::string model_out;
  std::string model_in;
  std::string pred_out;
  char *train_data;
  std::string task;

  int is_incre;
  int save_aux;
  int only_weight;

private:
  bool ParseLine(std::string& line, fea::instance& ins)
  {
    std::vector<std::string> fields;
    util::str_util::split(line, " ", fields);   
    ins.label = atoi(fields[0].c_str());
    for(size_t index = 1; index < fields.size(); ++index)
    {
      ins.fea_vec.push_back((uint32_t)(atoi(fields[index].c_str())));
    }   
    return true;
  }
  //predict one instance
  virtual ValueType PredIns(const fea::instance &ins)
  {
    ValueType sum = 0.0;
    std::vector<uint32_t> fea_vec = ins.fea_vec;
    for(size_t index = 0;index < fea_vec.size();++index)
    {
      uint32_t fea_index = fea_vec[index];
      if(model.count(fea_index))
        sum += model[fea_index].w;
    }
    return Sigmoid(sum);
  }

}; // end class FTRLSolver
} // end namespace algo
#endif

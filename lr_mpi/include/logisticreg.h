#ifndef __ADPREDICTALGO_LR_
#define __ADPREDICTALGO_LR_

#include <vector>
#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>

#include "io.h"

namespace adPredictAlgo {

struct LRModel {

public:
  LRModel():w(nullptr) {
  }

  virtual ~LRModel() {
    if(w != nullptr)
      delete [] w;
  }

  inline void Init() {
    
    if(w == nullptr)
      w = new float[num_fea];
    memset(w,0.0,num_fea);
  }

  inline void SetParam(const char *name,const char *val)
  {
    if(!strcmp(name,"num_fea"))
      num_fea = static_cast<size_t>(atoi(val));
  }

  float Sigmoid(float inx)
  {
    return 1. / (1 + std::exp(-inx));
  }

  float PredIns(const Instance &ins)
  {
    float inner = 0;
    for(size_t index = 0;index < ins.fea_vec.size();++index)
    {
      inner += w[ins.fea_vec[index]];
    }
    return Sigmoid(inner);
  }

  virtual void DumpModel(std::ofstream &os) {
    for(unsigned int i = 0;i < num_fea;++i)
    {
      os << w[i] << std::endl;
    }
  }

  virtual void LoadModel(std::ifstream &is) {
    is >> num_fea;

    w = new float[num_fea];
    for(unsigned int i = 0;i < num_fea;++i)
        is >> w[i];
  }

  float *w; // ffm model parameter
  size_t num_fea;
};

}//end class FFM

#endif //end namespace

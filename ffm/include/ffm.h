#ifndef __ADPREDICTALGO_FFM_
#define __ADPREDICTALGO_FFM_

#include <vector>
#include <dmlc/data.h>
#include <dmlc/io.h>

namespace adPredictAlgo {
//for sparse value
typedef struct {
  uint32_t fea_index; //feature index
  uint32_t field_index; //field index
}ffm_node;

typedef struct ins{
  int label;
  std::vector<ffm_node> fea_vec;

  ~ins(){}

  void clear(){
    label = 0;
    fea_vec.clear();
  }
}Instance;

class FFMModel {
public:
  FFMModel():w(nullptr) {
    n = 0;
    m = 0;
    d = 0;
    ffm_model_size = 0;
  }

  virtual ~FFMModel() {
    if(w != nullptr)
      delete [] w;
  }

  inline void Init() {
    CHECK(n != 0 || m != 0 || d != 0) << "the ffm parameter must be inital.";
    ffm_model_size = n + n * m * d;

    if(w == nullptr)
      w = new float[ffm_model_size];
  }

  inline void SetParam(const char *name,const char *val)
  {
    if(!strcmp(name,"num_fea"))
      n = static_cast<size_t>(atoi(val));
    if(!strcmp(name,"num_field"))
      m = static_cast<size_t>(atoi(val));
    if(!strcmp(name,"ffm_dim"))
      d = static_cast<size_t>(atoi(val));
  }

  inline float *operator [] (size_t i)
  {
    return (w + i);
  }

  virtual void DumpModel(const char *file) {

  }

  virtual void LoadModel(const char *file) {
  }

public:
  size_t n; // the number of feature
  size_t m; // the number of feild
  size_t d; // dim of the fm 
  size_t ffm_model_size;

  float *w; // ffm model parameter
  
};

}//end class FFM

#endif //end namespace

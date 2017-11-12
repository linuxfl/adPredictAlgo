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
  FFMModel():w(nullptr),v(nullptr) {
    n = 0;
    m = 0;
    d = 0;
  }

  virtual ~FFMModel() {
    if(w != nullptr)
      delete w;
    if(v != nullptr)
      delete v;
  }

  inline void Init() {
//    CHECK(n == 0 || m == 0 || d == 0) << "the ffm parameter must be inital."l
    if(w == nullptr)
      w = new float[n];
    if(v == nullptr)
      v = new float[n*m*d];
  }

  inline void SetParam(const char *name,const char *val)
  {
    if(!strcmp(name,"num_fea"))
      n = static_cast<size_t>(atoi(val));
    if(!strcmp(name,"num_feild"))
      m = static_cast<size_t>(atoi(val));
    if(!strcmp(name,"fm_dim"))
      d = static_cast<size_t>(atoi(val));
  }

public:
  size_t n; // the number of feature
  size_t m; // the number of feild
  size_t d; // dim of the fm

  float *w; // lr parameter
  float *v; // ffm parameter
  
};

}//end class FFM

#endif //end namespace

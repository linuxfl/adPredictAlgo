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

  virtual ~ins(){}
  
  void clear(){
    label = 0;
    fea_vec.clear();
  }
}Instance;

class FFMModel {
public:
  struct ModelParam {
    size_t n; // number of the feature
    size_t m; // number of the field
    size_t d; // number of the ffm dim

    ModelParam() {
      memset(this,0,sizeof(ModelParam));
    }
  };

  FFMModel():w(nullptr) {
    ffm_model_size = 0;
  }

  virtual ~FFMModel() {
    if(w != nullptr)
      delete [] w;
  }

  inline void Init() {
    CHECK(param.n != 0 || param.m != 0 || param.d != 0) << "the ffm parameter must be inital.";
    ffm_model_size = param.n + param.n * param.m * param.d + 1;

    if(w == nullptr)
      w = new float[ffm_model_size];
  }

  inline void SetParam(const char *name,const char *val)
  {
    if(!strcmp(name,"num_fea"))
      param.n = static_cast<size_t>(atoi(val));
    if(!strcmp(name,"num_field"))
      param.m = static_cast<size_t>(atoi(val));
    if(!strcmp(name,"ffm_dim"))
      param.d = static_cast<size_t>(atoi(val));
  }

  virtual void DumpModel(dmlc::Stream *fo) {
    fo->Write("binf",4);
    fo->Write(&param,sizeof(ModelParam));
    if(w != nullptr)
      fo->Write(w,sizeof(float) * ffm_model_size);
  }

  virtual void LoadModel(dmlc::Stream *fi) {
    std::string header;
    header.resize(4);
    CHECK(fi->Read(&header[0],4) != 0) << "invalid model.";
    if(header == "binf") {
      fi->Read(&param,sizeof(ModelParam));
      ffm_model_size = param.n * param.m * param.d + param.n + 1;
      if(w == nullptr) {
        w = new float[ffm_model_size];
        fi->Read(w,sizeof(float) * ffm_model_size);
      }
    }

  }

  size_t GetModelSize() const {
    return ffm_model_size;
  }

  ModelParam param;
  float *w; // ffm model parameter
private:
  size_t ffm_model_size;
};

}//end class FFM

#endif //end namespace

#ifndef __ADPREDICTALGO_FM_
#define __ADPREDICTALGO_FM_

#include <vector>
#include <dmlc/data.h>
#include <dmlc/io.h>
#include <fstream>

namespace adPredictAlgo {
//for sparse value
typedef float ValueType;

typedef struct ins{
  int label;
  std::vector<uint32_t> fea_vec;

  virtual ~ins(){}
  
  void clear(){
    label = 0;
    fea_vec.clear();
  }
}Instance;

class FMModel {
public:
  struct ModelParam {
    size_t n; // number of the feature
    size_t d; // number of the ffm dim

    ModelParam() {
      memset(this,0,sizeof(ModelParam));
    }
  };

  FMModel():w(nullptr) {
    fm_model_size = 0;
  }

  virtual ~FMModel() {
    if(w != nullptr)
      delete [] w;
  }

  inline void Init() {
    CHECK(param.n != 0 || param.d != 0) << "the fm parameter must be inital.";
    fm_model_size = param.n + param.n * param.d + 1;

    if(w == nullptr)
      w = new ValueType[fm_model_size];
  }

  inline void SetParam(const char *name,const char *val)
  {
    if(!strcmp(name,"num_fea"))
      param.n = static_cast<size_t>(atoi(val));
    if(!strcmp(name,"fm_dim"))
      param.d = static_cast<size_t>(atoi(val));
  }

  virtual void DumpModel(dmlc::Stream *fo) {
    fo->Write("binf",4);
    fo->Write(&param,sizeof(ModelParam));
    if(w != nullptr)
      fo->Write(w,sizeof(ValueType) * fm_model_size);
  }

  virtual void LoadModel(dmlc::Stream *fi) {
    std::string header;
    header.resize(4);
    CHECK(fi->Read(&header[0],4) != 0) << "invalid model.";
    if(header == "binf") {
      fi->Read(&param,sizeof(ModelParam));
      fm_model_size = param.n * param.d + param.n + 1;
      if(w == nullptr) {
        w = new ValueType[fm_model_size];
        fi->Read(w,sizeof(ValueType) * fm_model_size);
      }
    }

  }

  inline ValueType& w_0(){
    return *(w + fm_model_size - 1);
  }

  inline ValueType w_0() const
  {
    return *(w + fm_model_size - 1);
  }

  inline ValueType& w_i(uint32_t idx)
  {
    return *(w + idx);
  }

  inline ValueType w_i(uint32_t idx) const {
    return *(w + idx);
  }

  inline ValueType& W(uint32_t idx)
  {
    return *(w + idx);
  }

  inline ValueType W(uint32_t idx) const {
    return *(w + idx);
  }

  inline ValueType V(uint32_t idx) const{
    return *(w + param.n + idx);
  }

  inline ValueType& V(uint32_t idx){
    return *(w + param.n + idx);
  }

  virtual void DumpModel(std::ofstream &os) {
    os << param.n << " " << param.d << std::endl;
    for(unsigned int i = 0;i < fm_model_size;++i)
    {
      os << w[i] << std::endl;
    }
  }

  virtual void LoadModel(std::ifstream &is) {
    is >> param.n >> param.d;
    fm_model_size = param.n * param.d + param.n + 1;

    w = new ValueType[fm_model_size];
    for(unsigned int i = 0;i < fm_model_size;++i)
        is >> w[i];
  }

  size_t GetModelSize() const {
    return fm_model_size;
  }

  ModelParam param;
  ValueType *w; // fm model parameter
private:
  size_t fm_model_size;
};

}//end class FM

#endif //end namespace

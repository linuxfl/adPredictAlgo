#ifndef __ADPREDICTALGO_FTRL_TRAIN_
#define __ADPREDICTALGO_FTRL_TRAIN_

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>

#include "fm.h"
#include "str_util.h"
#include "metric.h"
#include "elapse.h"

namespace adPredictAlgo {

class FTRL {
public:
  FTRL(char *dtrain,dmlc::RowBlockIter<unsigned> *dtest)
    :dtrain(dtrain),dtest(dtest),n(nullptr),
    z(nullptr),n_fm(nullptr),z_fm(nullptr)
   {
    alpha = 0.01f;
    beta = 1.0f;
    alpha_fm = 0.01f;
    beta_fm = 1.0f;

    l1_reg = 0.0f;
    l2_reg = 0.1f;
    l1_fm_reg = 0.0f;
    l2_fm_reg = 0.1f;
    task = "train";
    model_out = "fm_model.dat";
    model_in = "NULL";
    num_epochs = 1;
  }

  virtual ~FTRL() 
  {
    if(z)
      delete [] z;
    if(n)
      delete [] n;
    if(z_fm)
      delete [] z_fm;
    if(n_fm)
      delete [] n_fm;
  }

  inline void Init()
  {
    fm.Init();
    size_t ftrl_param_size = fm.param.n * fm.param.d;
    fm_model_size = fm.GetModelSize();

    ValueType memory_use = (fm_model_size + ftrl_param_size) * sizeof(ValueType) * 1.0 / 1024 / 1024 / 1024;
    LOG(INFO) << "num_fea=" << fm.param.n << ", fm_dim=" << fm.param.d << ", num_epochs=" 
              << num_epochs << ", use_memory=" << memory_use << " GB";
    
    LOG(INFO) << "alpha=" << alpha << ", beta=" << beta << ", alpha_fm=" << alpha_fm << ", beta_fm=" << beta_fm
              << ", l1_reg=" << l1_reg << ", l2_reg=" << l2_reg << ", l1_fm_reg=" << l1_fm_reg << ", l2_fm_reg=" << l2_fm_reg;

    d_ = fm.param.d;
    n_ = fm.param.n;
    z = new ValueType[n_ + 1];
    n = new ValueType[n_ + 1];
    z_fm = new ValueType[ftrl_param_size];
    n_fm = new ValueType[ftrl_param_size];

    memset(z, 0.0, (n_ + 1) * sizeof(ValueType));
    memset(n, 0.0, (n_ + 1) * sizeof(ValueType));
    memset(n_fm, 0.0, (ftrl_param_size) * sizeof(ValueType));

    p_gauss_distribution = new std::normal_distribution<ValueType>(0.0,0.01);
    for(size_t i = 0;i < ftrl_param_size;++i){
      z_fm[i] = (*p_gauss_distribution)(generator);
    }
  }

  inline void SetParam(const char *name,const char *val) {
    if(!strcmp(name,"alpha"))
      alpha = static_cast<float>(atof(val));
    if(!strcmp(name,"beta"))
      beta = static_cast<float>(atof(val));
    if(!strcmp(name,"alpha_fm"))
      alpha_fm = static_cast<float>(atof(val));
    if(!strcmp(name,"beta_fm"))
      beta_fm = static_cast<float>(atof(val));
    if(!strcmp(name,"l1_reg"))
      l1_reg = static_cast<float>(atof(val));
    if(!strcmp(name,"l2_reg"))
      l2_reg = static_cast<float>(atof(val));
    if(!strcmp(name,"l1_fm_reg"))
      l1_fm_reg = static_cast<float>(atof(val));
    if(!strcmp(name,"l2_fm_reg"))
      l2_fm_reg = static_cast<float>(atof(val));
    if(!strcmp(name,"task"))
      task = val;
    if(!strcmp(name,"model_out"))
      model_out = val;
    if(!strcmp(name,"model_in"))
      model_in = val;
    if(!strcmp(name,"num_epochs"))
      num_epochs = static_cast<unsigned>(atoi(val));

    fm.SetParam(name,val);
  }

  inline void TaskTrain()
  {
    Timer t;
    t.Start();
    std::string line;
    Instance ins;
    for(unsigned i = 0;i < num_epochs;++i) {

      std::ifstream train_stream(dtrain);
      CHECK(train_stream.fail() == false) << "open the train file error!";

      int cnt = 0;

      while(getline(train_stream,line)) {
        ins.clear();
        ParseLine(line,ins);
        UpdateOneIter(ins);
        cnt++;
        if(cnt % 100000 == 0)
        {  
          TaskPred();
        }
      }
    }
    TaskPred();
    t.Stop();
    LOG(INFO) << "Elapsed time:" << t.ElapsedSeconds() << " sec.";
    DumpModel(model_out.c_str());
  }

    // 23:123444 v.index[j]:v.get_value(j) 
  inline ValueType PredIns(const dmlc::Row<unsigned> &v) {
    
    ValueType inner = fm.w_i(fm_model_size - 1);
    for(unsigned i = 0;i < v.length;++i)
    {
      uint32_t fid = v.index[i];
      inner += fm.w_i(fid);
    }

    for(size_t i = 0;i < v.length;++i)
    {
      uint32_t fea_x = v.index[i];
      uint32_t real_fea_x = n_ + fea_x * d_;

      for(size_t j = i+1;j < v.length;++j) {
        uint32_t fea_y = v.index[j];
        uint32_t real_fea_y = n_ + fea_y * d_;

        if(i!=j){
          for(size_t k = 0;k < d_;++k) {
            inner += fm.w_i(real_fea_x + k) * fm.w_i(real_fea_y + k);
          }
        }
      }
    }
    return Sigmoid(inner);
  }

  void TaskPred()
  {
      pair_vec.clear();
      dtest->BeforeFirst();
      while(dtest->Next()) {
        const dmlc::RowBlock<unsigned> &batch = dtest->Value();
        for(size_t i = 0;i < batch.size;i++) {
          dmlc::Row<unsigned> v = batch[i];
            ValueType score = PredIns(v);
            Metric::pair_t p(score,v.get_label());
            pair_vec.push_back(p);
        }
      }
      LOG(INFO) << "Test AUC=" << Metric::CalAUC(pair_vec) 
                << ",COPC=" << Metric::CalCOPC(pair_vec);
  }

  virtual void ParseLine(const std::string &line, Instance &ins)
  {
    std::vector<std::string> fea_vec;
    util::str_util::split(line, " ", fea_vec);
    ins.label = static_cast<int>(atoi(fea_vec[0].c_str()));

    for(size_t i = 1;i < fea_vec.size();i++) {
      uint32_t fid = 
            static_cast<uint32_t>(atoi(fea_vec[i].c_str()));
      ins.fea_vec.push_back(fid);
    }
  }

  virtual ValueType PredictRaw(Instance &ins)
  {
    size_t ins_len = ins.fea_vec.size();
    std::vector<uint32_t> &fea_vec = ins.fea_vec;
    ValueType sum = 0.0;
    //w_0 update
    fm.w_i(fm_model_size - 1) = ( - z[n_]) / \
                ((beta + std::sqrt(n[n_])) / alpha);
    sum += fm.w_i(fm_model_size - 1);

    //w_i update
    for(size_t idx = 0;idx < ins_len;++idx)
    {
      uint32_t fid = fea_vec[idx];
      if(std::fabs(z[fid]) < l1_reg) {
        fm.w_i(fid) = 0.0;
      }else{
        fm.w_i(fid) = (Sign(z[fid]) * l1_reg - z[fid]) / \
                      ((beta + std::sqrt(n[fid])) / alpha + l2_reg);
      }
      sum += fm.w_i(fid);
    }

    //v_i update
    for(size_t i = 0;i < ins_len;++i) {
      uint32_t fea_x = fea_vec[i];
      for(size_t k = 0;k < d_;++k) {
        uint32_t real_fid = fea_x * d_ + k;
        uint32_t map_fid = real_fid + n_;

        if(std::fabs(z_fm[real_fid]) < l1_fm_reg){
          fm.w_i(map_fid) = 0.0;
        }else{
          fm.w_i(map_fid) = (Sign(z_fm[real_fid]) * l1_fm_reg - z_fm[real_fid]) / \
                          ((beta_fm + std::sqrt(n_fm[real_fid])) / alpha_fm + l2_fm_reg);
        }
        sum += fm.w_i(map_fid);
      }
    }

    return sum;
  }

  virtual ValueType Predict(ValueType inx) {
    return Sigmoid(inx);
  }

  inline int Sign(ValueType inx)
  {
    return inx > 0?1:0;
  }

  inline ValueType Sigmoid(ValueType inx)
  {
    CHECK(!std::isnan(inx)) << "nan occurs";
    ValueType tuc_val = 31;
    return 1. / (1. + std::exp(-std::max(std::min(inx,tuc_val),-tuc_val)));
  }

  virtual void AuxUpdate(const Instance &ins,ValueType grad)
  {
    size_t ins_len = ins.fea_vec.size();
    std::vector<uint32_t> fea_vec = ins.fea_vec;

    ValueType sigma = (std::sqrt(n[n_] + grad * grad) - std::sqrt(n[n_])) / alpha;
    z[n_] += grad - sigma * fm.w_i(fm_model_size - 1);
    n[n_] += grad * grad;

    for(size_t idx = 0;idx < ins_len;++idx)
    {
      uint32_t fid = fea_vec[idx];
      ValueType theta = (std::sqrt(n[fid] + grad * grad) - std::sqrt(n[fid])) / alpha;
      z[fid] += grad - theta * fm.w_i(fid);
      n[fid] += grad * grad;
    }
    std::unordered_map<uint32_t, ValueType> sum_fm;

    for(size_t i = 0;i < ins_len;++i)
    {
      uint32_t fea_x = fea_vec[i];
      uint32_t real_fea_x = fea_x * d_ + n_;

      for(size_t j = 0; j < ins_len;++j)
      {
        if(i != j) {
          uint32_t fea_y = fea_vec[j];
          uint32_t real_fea_y = fea_y * d_ + n_;

          for(size_t k = 0;k < d_;++k) {
            uint32_t real_fid = real_fea_x + k;
            if(sum_fm.find(real_fid) != sum_fm.end())
              sum_fm[real_fid] += fm.w_i(real_fea_y + k);
            else
              sum_fm[real_fid] = fm.w_i(real_fea_y + k);
          }
        }
      }
    }

    for(size_t i = 0;i < ins_len;++i) {
      uint32_t fea_x = fea_vec[i];
      for(size_t j = 0;j < ins_len;++j) {
        if(i != j){
          for(size_t k = 0;k < fm.param.d;++k){
            uint32_t real_fid = fea_x * d_ + k;
            uint32_t map_fid = real_fid + n_;
            ValueType g_fm = grad * sum_fm[map_fid];
            ValueType theta = (std::sqrt(n_fm[real_fid] + g_fm * g_fm) - std::sqrt(n_fm[real_fid])) / alpha_fm;
            z_fm[real_fid] += g_fm - theta * fm.w_i(map_fid);
            n_fm[real_fid] += g_fm * g_fm;
          }
        }
      }
    }
  }

  virtual void UpdateOneIter(Instance &ins)
  {
    ValueType p = Predict(PredictRaw(ins));
    int label = ins.label;
    ValueType grad = p - label;
    AuxUpdate(ins,grad);
  }

  virtual void Run()
  {
    if(task == "train") {    
      this->Init();
      this->TaskTrain();
    }else if(task == "pred") {
      LOG(INFO) << "Load FM Model now...";
      this->LoadModel(model_in.c_str());
      this->TaskPred();
    }else
      LOG(FATAL) << "error task!";
  }

  virtual void DumpModel(const char *model_out) {
    dmlc::Stream *fo = dmlc::Stream::Create(model_out,"w");
    fm.DumpModel(fo);
    delete fo;
//    std::ofstream os(model_out);
//    CHECK(os.fail() == false) << "open model out error!";

//    ffm.DumpModel(os);
//    os.close();
  }

  virtual void LoadModel(const char *model_in) {
    dmlc::Stream *fi = dmlc::Stream::Create(model_in,"r");
//    std::ifstream is(model_in);
//    CHECK(is.fail() == false) << "open model in error!";
    fm.LoadModel(fi);
    delete fi;
//    is.close();
  }
private:
  char *dtrain;
  dmlc::RowBlockIter<unsigned> *dtest;

  FMModel fm;
  ValueType *n,*z;
  ValueType *n_fm,*z_fm;

  float alpha,beta;
  float alpha_fm,beta_fm;

  float l1_reg,l2_reg;
  float l1_fm_reg,l2_fm_reg;

  uint32_t d_,n_; // number of feature and dim
  std::string task;
  std::string model_out;
  std::string model_in;

  unsigned num_epochs;
  size_t fm_model_size;

  std::vector<Metric::pair_t> pair_vec;
  std::default_random_engine generator;
  std::normal_distribution<ValueType> *p_gauss_distribution;
}; //end class

} // end namespace
#endif

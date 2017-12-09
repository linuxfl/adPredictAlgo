#ifndef __ADPREDICTALGO_FTRL_TRAIN_
#define __ADPREDICTALGO_FTRL_TRAIN_

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>

#include "ffm.h"
#include "str_util.h"
#include "metric.h"
#include "elapse.h"

namespace adPredictAlgo {

class FTRL {
public:
  FTRL(char *dtrain,dmlc::RowBlockIter<unsigned> *dtest)
    :dtrain(dtrain),dtest(dtest),n(nullptr),
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
    task = "train";
    model_out = "ffm_model.dat";
    model_in = "NULL";
    num_epochs = 1;
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
    size_t ftrl_param_size = ffm.param.n * ffm.param.m * ffm.param.d;
    ffm_model_size = ffm.GetModelSize();

    double memory_use = (ffm.param.n * 3 + ftrl_param_size * 3) * sizeof(double) * 1.0 / 1024 / 1024 / 1024;
    LOG(INFO) << "num_fea=" << ffm.param.n << ", ffm_dim=" << ffm.param.d << ", num_field="<< ffm.param.m 
              << ", num_epochs=" << num_epochs << ", use_memory=" << memory_use << " GB";
    
    LOG(INFO) << "alpha=" << alpha << ", beta=" << beta << ", alpha_ffm=" << alpha_ffm << ", beta_ffm=" << beta_ffm
              << ", l1_reg=" << l1_reg << ", l2_reg=" << l2_reg << ", l1_ffm_reg=" << l1_ffm_reg << ", l2_ffm_reg=" << l2_ffm_reg;

    z = new double[ffm.param.n + 1];
    n = new double[ffm.param.n + 1];
    z_ffm = new double[ftrl_param_size];
    n_ffm = new double[ftrl_param_size];

    p_gauss_distribution = new std::normal_distribution<double>(0.0,0.01);
    for(size_t i = 0;i < ftrl_param_size;++i){
      z_ffm[i] = (*p_gauss_distribution)(generator);
    }
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
    if(!strcmp(name,"task"))
      task = val;
    if(!strcmp(name,"model_out"))
      model_out = val;
    if(!strcmp(name,"model_in"))
      model_in = val;
    if(!strcmp(name,"num_epochs"))
      num_epochs = static_cast<unsigned>(atoi(val));

    ffm.SetParam(name,val);
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
  inline double PredIns(const dmlc::Row<unsigned> &v) {
    double inner = ffm.w[ffm_model_size - 1];
    for(unsigned i = 0;i < v.length;++i)
    {
      uint32_t fea_index = v.get_value(i);
      inner += ffm.w[fea_index];
    }

    for(size_t i = 0;i < v.length;++i)
    {
      uint32_t fea_index = v.get_value(i);
      uint32_t field_index = v.index[i];
      for(size_t k = 0;k < ffm.param.d;++k) {
        uint32_t map_fea_index = 
                  ffm.param.n + (fea_index) * ffm.param.m * ffm.param.d + (field_index - 1) * ffm.param.d + k;
        inner += ffm.w[map_fea_index];
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
            double score = PredIns(v);
            Metric::pair_t p(score,v.get_label());
            pair_vec.push_back(p);
        }   
      }   
      LOG(INFO) << "Test AUC=" << Metric::CalAUC(pair_vec) 
                << ",COPC=" << Metric::CalCOPC(pair_vec);
  }   

  virtual void ParseLine(const std::string &line,Instance &ins)
  {
    std::vector<std::string> fea_vec;
    util::str_util::split(line," ",fea_vec);
    ins.label = static_cast<int>(atoi(fea_vec[0].c_str()));

    for(size_t i = 1;i < fea_vec.size();i++) {
      std::vector<std::string> kvs;
      ffm_node _node;
      util::str_util::split(fea_vec[i],":",kvs);

      _node.field_index = static_cast<uint32_t>(atoi(kvs[0].c_str()));
      _node.fea_index = static_cast<uint32_t>(atoi(kvs[1].c_str()));

      CHECK(_node.field_index <= ffm.param.m) << "field index must less then the number of field.";
      CHECK(_node.fea_index < ffm.param.n) << "fea index must less then the number of fea";

      ins.fea_vec.push_back(_node);
    }
  }

  virtual double PredictRaw(const Instance &ins)
  {
    size_t ins_len = ins.fea_vec.size();
    std::vector<ffm_node> fea_vec = ins.fea_vec;
    double sum = 0.0f;
    //w_0 update
    if(std::fabs(z[ffm.param.n]) < l1_reg) {
      ffm.w[ffm_model_size - 1] = 0.0;
    }else{
      ffm.w[ffm_model_size - 1] = (Sign(z[ffm.param.n]) * l1_reg - z[ffm.param.n]) / \
                ((beta + std::sqrt(n[ffm.param.n])) / alpha + l2_reg);
    }
    sum += ffm.w[ffm_model_size - 1];    
    //w_i update
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

    //v_i_f update
    for(size_t index = 0;index < ins_len;++index)
    {
      uint32_t fea_index = fea_vec[index].fea_index;
      uint32_t field_index = fea_vec[index].field_index;
      for(size_t k = 0;k < ffm.param.d;++k) {
        uint32_t real_fea_index = 
                  (fea_index) * ffm.param.m * ffm.param.d + (field_index - 1) * ffm.param.d + k;
        uint32_t map_fea_index = real_fea_index + ffm.param.n;

        if(std::fabs(z_ffm[real_fea_index]) < l1_ffm_reg){
          ffm.w[map_fea_index] = 0.0f;
        }else{
          ffm.w[map_fea_index] = (Sign(z_ffm[real_fea_index]) * l1_ffm_reg - z_ffm[real_fea_index]) / \
                          ((beta_ffm + std::sqrt(n_ffm[real_fea_index])) / alpha_ffm + l2_ffm_reg);
        }
        sum += ffm.w[map_fea_index];
      }
    }
    return sum;
  }
  
  virtual double Predict(double inx) {
    return Sigmoid(inx);
  }

  inline int Sign(double inx)
  {
    return inx > 0?1:0;
  }

  inline double Sigmoid(double inx)
  {
    CHECK(!std::isnan(inx)) << "nan occurs";
    return 1. / (1. + std::exp(-std::max(std::min(inx,31.),-31.)));
  }

  virtual void AuxUpdate(const Instance &ins,double grad)
  {
    size_t ins_len = ins.fea_vec.size();
    std::vector<ffm_node> fea_vec = ins.fea_vec;

    double sigma = (std::sqrt(n[ffm.param.n] + grad * grad) - std::sqrt(n[ffm.param.n])) / alpha;
    z[ffm.param.n] += grad - sigma * ffm.w[ffm_model_size - 1];
    n[ffm.param.n] += grad * grad;

    for(size_t index = 0;index < ins_len;++index)
    {
      uint32_t fea_index = fea_vec[index].fea_index;
      double theta = (std::sqrt(n[fea_index] + grad * grad) - std::sqrt(n[fea_index])) / alpha;
      z[fea_index] += grad - theta * ffm.w[fea_index];
      n[fea_index] += grad * grad;
    }
    std::map<uint32_t,double> sum_ffm;
    for(size_t i = 0;i < ins_len;++i)
    {
      uint32_t fea_x = fea_vec[i].fea_index;
      uint32_t field_x = fea_vec[i].field_index;
      uint32_t real_fea_x = 
                  ffm.param.n + (fea_x) * ffm.param.m * ffm.param.d + (field_x - 1) * ffm.param.d;

      for(size_t j = 0; j < ins_len;++j)
      {
        uint32_t fea_y = fea_vec[j].fea_index;
        uint32_t field_y = fea_vec[j].field_index;
        uint32_t real_fea_y = 
                  ffm.param.n + (fea_y) * ffm.param.m * ffm.param.d + (field_y - 1) * ffm.param.d;
        if(i != j) {
          for(size_t k = 0;k < ffm.param.d;++k) {
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
      for(size_t k = 0;k < ffm.param.d;++k) {
        uint32_t real_fea_index = 
                  (fea_index) * ffm.param.m * ffm.param.d + (field_index - 1) * ffm.param.d + k;
        uint32_t map_fea_index = real_fea_index + ffm.param.n;
        double g_ffm = grad * sum_ffm[map_fea_index];
        double theta = (std::sqrt(n_ffm[real_fea_index] + g_ffm * g_ffm) - std::sqrt(n_ffm[real_fea_index]));
        z_ffm[real_fea_index] += g_ffm - theta * ffm.w[map_fea_index];
        n_ffm[real_fea_index] += g_ffm * g_ffm;
      }
    }
  }


  virtual void UpdateOneIter(const Instance &ins) 
  {
    double p = Predict(PredictRaw(ins));
    int label = ins.label;
    double grad = p - label;
    AuxUpdate(ins,grad);
  }

  virtual void Run()
  {
    if(task == "train") {    
      this->Init();
      this->TaskTrain();
    }else if(task == "pred") {
      LOG(INFO) << "Load FFM Model now...";
      this->LoadModel(model_in.c_str());
      this->TaskPred();
    }else
      LOG(FATAL) << "error task!";
  }

  virtual void DumpModel(const char *model_out) {
//    dmlc::Stream *fo = dmlc::Stream::Create(model_out,"w");
//    ffm.DumpModel(fo);
//    delete fo;
    std::ofstream os(model_out);
    CHECK(os.fail() == false) << "open model out error!";

    ffm.DumpModel(os);
    os.close();
  }

  virtual void LoadModel(const char *model_in) {
    dmlc::Stream *fi = dmlc::Stream::Create(model_in,"r");
//    std::ifstream is(model_in);
//    CHECK(is.fail() == false) << "open model in error!";
    ffm.LoadModel(fi);
    delete fi;
//    is.close();
  }
private:
  char *dtrain;
  dmlc::RowBlockIter<unsigned> *dtest;

  FFMModel ffm;
  double *n,*z;
  double *n_ffm,*z_ffm;

  float alpha,beta;
  float alpha_ffm,beta_ffm;

  float l1_reg,l2_reg;
  float l1_ffm_reg,l2_ffm_reg;

  std::string task;
  std::string model_out;
  std::string model_in;

  unsigned num_epochs;
  size_t ffm_model_size;

  std::vector<Metric::pair_t> pair_vec;
  std::default_random_engine generator;
  std::normal_distribution<double> *p_gauss_distribution;
}; //end class

} // end namespace
#endif

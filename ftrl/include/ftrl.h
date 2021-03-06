#ifndef _FTRL_H_
#define _FTRL_H_

#include <iostream>
#include <dmlc/io.h>
#include <dmlc/data.h>
#include <cstring>
#include <fstream>
#include <algorithm>

#include "str_util.h"
#include "metric.h"

namespace adPredictAlgo{

//instance struct
typedef struct {
  int label;
  std::vector<unsigned> fea_vec;
  void reset() {
    label = 0;
    fea_vec.clear();
  }
}instance;
class Ftrl
{
  public:
    float l2_reg; //l2 norm
    float l1_reg; //l1 norm
    float alpha;  //ftrl parameter
    float beta;   //ftrl parameter
    float base_score;

    char *traind;  
    dmlc::RowBlockIter<unsigned> *dtrain;
    dmlc::RowBlockIter<unsigned> *dtest;
     
    Ftrl(dmlc::RowBlockIter<unsigned> *dtrain,
          dmlc::RowBlockIter<unsigned> *dtest):
    dtrain(dtrain),dtest(dtest),
    w(nullptr),n(nullptr),z(nullptr)
    {
      l2_reg = 0.1f;
      l1_reg = 0.0f;
      alpha = 0.01f;
      beta = 1.0f;
      base_score = 0.5f;

      num_fea = 0;
      model_in = "NULL";
      model_out = "lr_model.dat";
      memory_in = "batch";
      pred_out = "pred.txt";
      epoch = 1;
    }

     Ftrl(char *traind,dmlc::RowBlockIter<unsigned> *dtest):
        traind(traind),dtest(dtest),
        w(nullptr),n(nullptr),z(nullptr)
    {
      l2_reg = 0.1f;
      l1_reg = 0.0f;
      alpha = 0.01f;
      beta = 1.0f;
      base_score = 0.5f;

      num_fea = 0;
      model_in = "NULL";
      model_out = "lr_model.dat";
      memory_in = "stream";
      pred_out = "pred.txt";
      dtrain = nullptr;
      epoch = 1;
    }

    virtual ~Ftrl() {
      
      if(dtrain != nullptr) 
        delete dtrain;
      delete dtest;
      
      if(w != nullptr) delete [] w;
      if(n != nullptr) delete [] n;
      if(z != nullptr) delete [] z;
    }
    
    inline void InitBaseScore() {
      CHECK(base_score > 0.0f &&  base_score < 1.0f) << 
        "base score must be in (0,1) for logistic loss";
      base_score = -std::log(1.0f/base_score - 1.0f);
    }

    inline void SetParam(const char *name,const char *val) {
      if (!strcmp(name,"model_in")) 
        model_in = val;
      if (!strcmp(name,"model_out")) 
        model_out = val;
      if (!strcmp(name,"l1_reg")) 
        l1_reg = static_cast<float>(atof(val));
      if (!strcmp(name,"l2_reg")) 
        l2_reg = static_cast<float>(atof(val));
      if (!strcmp(name,"alpha")) 
        alpha = static_cast<float>(atof(val));
      if (!strcmp(name,"beta")) 
        beta = static_cast<float>(atof(val));
      if (!strcmp(name,"num_fea")) 
        num_fea = static_cast<size_t>(atoi(val));
      if (!strcmp(name,"base_score")) 
        base_score = static_cast<float>(atof(val));
      if (!strcmp(name,"pred_out"))
        pred_out = val;
      if (!strcmp(name,"epoch")) 
        epoch = static_cast<int>(atoi(val));
    }

    inline void Run()
    {
      this->Init();
      if(model_in != "NULL")
      {
        this->LoadModel(model_in.c_str());
      }
      if (memory_in == "stream") {
        this->TrainOnStream();
      }else {
        this->TrainOnBatch();
      }
      LOG(INFO) << "finish train,save model now...";
      this->SaveModel(model_out.c_str());
    }

    inline void Init()
    {
      CHECK(num_fea > 0) << "num_fea get error!please check your data!";
      if(dtest->NumCol() > num_fea)
        LOG(INFO) << "the number feature of test data is bigger then the parameter!";

      LOG(INFO) << "num_fea=" << num_fea << ",alpha="
                << alpha << ",beta=" << beta << ",l1_reg=" << l1_reg
                << ",l2_reg=" << l2_reg << ",base_score=" << base_score;

      //init weight and ftrl paramter
      w = new float[num_fea];
      n = new float[num_fea];
      z = new float[num_fea];

      memset(w,0.0,sizeof(float) * num_fea);
      memset(n,0.0,sizeof(float) * num_fea);
      memset(z,0.0,sizeof(float) * num_fea);
      InitBaseScore();
    }

    void TrainOnBatch() {
      unsigned count = 0;
      dtrain->BeforeFirst();
      while(dtrain->Next())  {
        const  dmlc::RowBlock<unsigned> &batch = dtrain->Value();
        for(size_t i = 0;i < batch.size;i++)
        {
          dmlc::Row<unsigned> v = batch[i];
          for(unsigned j = 0;j < v.length;j++) {
            unsigned fea_index = v.index[j];
            float z_val = z[fea_index];
            float n_val = n[fea_index];

            if (fabs(z_val) < l1_reg) {
              w[fea_index] = 0;
            }else{
              w[fea_index] = (Sign(z_val) * l1_reg - z_val) / \
                (l2_reg + (beta + std::sqrt(n_val)) / alpha);
            }
          }
          float grad = PredIns(v) - v.get_label();
          for(unsigned j = 0;j < v.length;j++) {
            unsigned fea_index = v.index[j];
            float w_val = w[fea_index];
            float n_val = n[fea_index];
            
            float theta = (std::sqrt(n_val + grad * grad) \
                - std::sqrt(n_val)) / alpha;
            z[fea_index] += grad - theta * w_val;
            n[fea_index] += grad * grad;
          }
          
          count++;
          if (count % 100000 == 0) {
//            LOG(INFO) << "train instance : " << count;
            TaskPred();
          }
        }
      }
      TaskPred();
      PredOut();
    }

    inline void ParseLine(const std::string line,instance &ins) {
      std::vector<std::string> fields;
      util::str_util::split(line," ",fields);
      ins.label = static_cast<int>(atoi(fields[0].c_str()));
      for(size_t index = 1;index < fields.size();++index) {
        ins.fea_vec.push_back(static_cast<unsigned>(atoi(fields[index].c_str())));
      }
    }

    void TrainIns(const instance &ins) {
      std::vector<unsigned> fea_vec = ins.fea_vec;
      size_t ins_len = fea_vec.size();
      for(size_t i = 0;i < ins_len;i++)
      {
        unsigned fea_index = fea_vec[i];
        float z_val = z[fea_index];
        float n_val = n[fea_index];

        if (fabs(z_val) < l1_reg) {
          w[fea_index] = 0;
         }else{
          w[fea_index] = (Sign(z_val) * l1_reg - z_val) / \
            (l2_reg + (beta + std::sqrt(n_val)) / alpha);
          }
      }
      float grad = PredIns(ins) - ins.label;
      for(size_t i = 0;i < ins_len;i++) {
        unsigned fea_index = fea_vec[i];
        float w_val = w[fea_index];
        float n_val = n[fea_index];
            
        float theta = (std::sqrt(n_val + grad * grad) \
            - std::sqrt(n_val)) / alpha;
        z[fea_index] += grad - theta * w_val;
        n[fea_index] += grad * grad;
       }
    }

    void TrainOnStream() {
      unsigned count = 0;
      instance ins;
      std::string line;
      for(int e = 0;e < epoch;++e) {
        count = 0;
        std::ifstream train_stream(traind);
        CHECK(train_stream.fail() == false)
            << "open the train data file error!";

        while(getline(train_stream,line)) {
          ins.fea_vec.clear();
          ParseLine(line,ins);
          TrainIns(ins);
          count++;
          if (count % 100000 == 0) {
            TaskPred();
          }
        }
        train_stream.close();
      }
      TaskPred();
      PredOut();
    }

    virtual void LoadModel(const char *model_in) {
      LOG(INFO) << "Load old model...";
      std::ifstream ifs(model_in);
      CHECK(ifs.fail() == false) << "open model_in error!";
      
      ifs >> alpha;
      ifs >> beta;
      ifs >> l1_reg;
      ifs >> l2_reg;
    
      unsigned i = 0;
      while(!ifs.eof()) {
        ifs >> n[i] >> z[i] >> w[i];
        i++;
      }
      ifs.close();
    }

    virtual void SaveModel(const char *model_out) {
      std::ofstream ofs(model_out);
      ofs << alpha << std::endl;
      ofs << beta << std::endl;
      ofs << l1_reg << std::endl;
      ofs << l2_reg << std::endl;

      for(size_t i = 0;i < num_fea;i++) {
        ofs << n[i] << " "
            << z[i] << " " 
            << w[i] << " "
            << std::endl;
      }
      ofs.close();
    }
    
    virtual void PredOut() {
      dmlc::Stream *fo = dmlc::Stream::Create(pred_out.c_str(),"w");
      dmlc::ostream os(fo);
      for(auto iter = pair_vec.begin();iter != pair_vec.end();iter++) {
        os << iter->score << "\n";
      }
      os.set_stream(NULL);
      delete fo;
    }

    void TaskPred() {
      pair_vec.clear();
      dtest->BeforeFirst();
      while(dtest->Next()) {
        const dmlc::RowBlock<unsigned> &batch = dtest->Value();
        for(size_t i = 0;i < batch.size;i++) {
          dmlc::Row<unsigned> v = batch[i];
          float score = PredIns(v);
          Metric::pair_t p(score,v.get_label());
          pair_vec.push_back(p);
        }
      }
      LOG(INFO) << "Test AUC=" << Metric::CalAUC(pair_vec) 
                << ",COPC=" << Metric::CalCOPC(pair_vec)
                << ",LogLoss=" << Metric::CalLogLoss(pair_vec)
                << ",MSE=" << Metric::CalMSE(pair_vec);
    }
    
    inline int Sign(float val) {
      return val > 0.0f?1:-1;
    }

    inline float Sigmoid(float inx) {
      return 1.0f / (1.0f + std::exp(-inx));    
    }
  
    inline float PredIns(const dmlc::Row<unsigned> &v) {
      float inner = 0.0f;
      for(unsigned i = 0;i < v.length;i++) {
        inner += w[v.index[i]];
      }
      inner += base_score;
      return Sigmoid(inner);
    }

    inline float PredIns(const instance &ins) {
      float inner = 0.0f;
      std::vector<unsigned> fea_vec = ins.fea_vec;
      size_t inslen = fea_vec.size();
      for(size_t i = 0;i < inslen;i++) {
        inner += w[fea_vec[i]];
      }
      inner += base_score;
      return Sigmoid(inner);
    }  

  private:
    float *w,*n,*z;
    size_t num_fea;
    std::vector<Metric::pair_t> pair_vec;

    std::string model_in;
    std::string model_out;
    std::string memory_in;
    std::string pred_out;
    int epoch;
};
}
#endif

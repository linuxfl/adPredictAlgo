#ifndef _ADPREDICT_ADMM_H_
#define _ADPREDICT_ADMM_H_

#include <iostream>
#include <fstream>
#include <math.h>

#include "dmlc/data.h"
#include "dmlc/io.h"
#include "learner.h"
#include "metric.h"

namespace adPredictAlgo {

const double RELTOL = 1e-4;
const double ABSTOL = 1e-5;

class ADMM {
  public:
   ADMM()
      :dtrain(nullptr),primal(nullptr),dual(nullptr)
      ,cons(nullptr),w(nullptr),cons_pre(nullptr)
   {
     learner = "NULL";
     num_fea = 0;
     num_data = 0;
     num_procs = 0;
     rank = -1;
    
     l1_reg = 0.0f;
     l2_reg = 0.0f;
     rho = 0.0f;

     admm_max_iter = 5;
     train_data = "NULL";
     pred_out = "pred.txt";
     model_in = "NULL";
     model_out = "lr_model.dat";
   }

   virtual ~ADMM() {
     if(primal != nullptr)
       delete [] primal;
     if(dual != nullptr)
       delete [] dual;
     if(cons != nullptr)
       delete [] cons;
     if(cons_pre != nullptr)
       delete [] cons_pre;
     if(w != nullptr)
       delete [] w;

     if(optimizer != nullptr)
       delete optimizer;
   }

   //init
   inline void Init() {
     CHECK(num_fea != 0 || learner != "NULL" || train_data != "NULL") 
           << "num_fea and name_leaner must be set!";
      //init paramter
     primal = new float[num_fea];
     dual = new float[num_fea];
     cons = new float[num_fea];
     cons_pre = new float[num_fea];
     w = new float[num_fea];

     memset(primal,0.0,sizeof(float) * num_fea);
     memset(dual,0.0,sizeof(float) * num_fea);
     memset(cons,0.0,sizeof(float) * num_fea);
   }
 
   void Configure(
      std::vector<std::pair<std::string,std::string> > cfg)
   {
      for(const auto &kv : cfg) {
        cfg_[kv.first] = kv.second;
      }

      if(cfg_.count("rank"))
        rank = static_cast<int>(atoi(cfg_["rank"].c_str()));
      if(cfg_.count("num_fea"))
        num_fea = static_cast<uint32_t>(atoi(cfg_["num_fea"].c_str()));
      if(cfg_.count("learner"))
        learner = cfg_["learner"];
      if(cfg_.count("admm_max_iter"))
        admm_max_iter = static_cast<int>(atoi(cfg_["admm_max_iter"].c_str()));
      if(cfg_.count("l1_reg"))
        l1_reg = static_cast<float>(atof(cfg_["l1_reg"].c_str()));
      if(cfg_.count("l2_reg"))
        l2_reg = static_cast<float>(atof(cfg_["l2_reg"].c_str()));
      if(cfg_.count("rho"))
        rho = static_cast<float>(atof(cfg_["rho"].c_str()));
      if(cfg_.count("train_data"))
        train_data = cfg_["train_data"];
      if(cfg_.count("num_procs"))
        num_procs = static_cast<int>(atoi(cfg_["num_procs"].c_str()));
      if(cfg_.count("pred_out"))
        pred_out = cfg_["pred_out"];
      if(cfg_.count("model_out"))
        model_out = cfg_["model_out"];
      if(cfg_.count("model_in"))
        model_in = cfg_["model_in"];

      CHECK(train_data != "NULL") << "train data must be set.";

      train_data += std::to_string(rank);
      dtrain = dmlc::RowBlockIter<unsigned>::Create(train_data.c_str(),0,1,"libsvm");

      dtrain->BeforeFirst();
      while(dtrain->Next())  {
        const dmlc::RowBlock<unsigned> &batch = dtrain->Value();
        num_data += batch.size;
      }
      if (rank == 0)
        LOG(INFO) << "num_fea=" << num_fea << ",num_data=" << num_data
                 << ",l1_reg=" << l1_reg << ",l2_reg=" << l2_reg 
                 << ",learner=" << learner << ",admm_max_iter=" << admm_max_iter
                 << ",rho=" << rho << ",train_data=" << train_data;

      //optimizer
      optimizer = Learner::Create(learner.c_str());
      if(optimizer == nullptr)
        LOG(FATAL) << "learner inital error!";
      optimizer->Configure(cfg);
   }

   void UpdatePrimal()
   {
      optimizer->Train(primal,dual,cons,rho,dtrain);
   }

   //update dual parameter y
   void UpdateDual() {
     for(uint32_t i = 0;i < num_fea;++i) {
       dual[i] += rho * (primal[i] - cons[i]);
     }
   }

   //update z
   void UpdateConsensus() {
     float s = 1. / (rho * num_procs + 2 * l2_reg);
     float kappa = s * l1_reg;
     for(uint32_t i = 0;i < num_fea;i++)
     {
       w[i] = s * (rho * primal[i] + dual[i]);
       cons_pre[i] = cons[i];
     }
 
     MPI_Allreduce(w, cons,  num_fea, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

     if(l1_reg != 0.0f)
       SoftThreshold(kappa,cons);
   }

   //soft threshold
   void SoftThreshold(float t_,float *z) {
     for(uint32_t i = 0;i < num_fea;i++){
       if(z[i] > t_){
         z[i] -= t_;
       }else if(z[i] <= t_ && z[i] >= -t_){
         z[i] = 0.0;
       }else{
         z[i] += t_;
       }
     }
   }

   //train task
   void TaskTrain() {
     int iter = 0;
     while(iter < admm_max_iter) {
       this->UpdatePrimal();
       this->UpdateConsensus();
       this->UpdateDual();
       if(IsStop(iter))
         break;
       iter++;
       //TaskPred();
     }
     SaveModel();
   }

   float Eva()
   {
     float sum = 0;
     dtrain->BeforeFirst();
     while(dtrain->Next())
     {
       const dmlc::RowBlock<unsigned> &batch = dtrain->Value();
       for(size_t i = 0;i < batch.size;++i)
        {
          dmlc::Row<unsigned> v = batch[i];
          if(v.get_label() == 0.0)
            sum += std::log(1 - optimizer->PredIns(v,cons));
          else
            sum += std::log(optimizer->PredIns(v,cons));
        }
      }
    return -sum;
   }

   //predict task
   void TaskPred() {
     pair_vec.clear();
      dtrain->BeforeFirst();
      while(dtrain->Next()) {
        const dmlc::RowBlock<unsigned> &batch = dtrain->Value();
        for(size_t i = 0;i < batch.size;i++) {
          dmlc::Row<unsigned> v = batch[i];
          float score = optimizer->PredIns(v,cons);
          Metric::pair_t p(score,v.get_label());
          pair_vec.push_back(p);
        }
      }
      LOG(INFO) << "Test AUC=" << Metric::CalAUC(pair_vec) 
                << ",COPC=" << Metric::CalCOPC(pair_vec)
                << ",LogLoss=" << Metric::CalLogLoss(pair_vec)
                << ",RMSE=" << Metric::CalMSE(pair_vec)
                << ",MAE=" << Metric::CalMAE(pair_vec);
 
      std::ofstream os(pred_out.c_str());
      for(size_t i = 0;i < pair_vec.size();i++)
        os << pair_vec[i].t_label << " " << pair_vec[i].score << std::endl;
      os.close();
   }

   //save model
   void SaveModel() {
     std::ofstream os(model_out.c_str());
     for(uint32_t i = 0;i < num_fea;i++)
       os << cons[i] << std::endl;
    os.close();
   }
   
   void LoadModel() {
    std::ifstream is(model_in.c_str());
    for(uint32_t i = 0;i < num_fea;i++)
      is >> cons[i];
    is.close();
   }

   bool IsStop(int iter) {
     float send[3] = {0};
     float recv[3] = {0};
  
     for(uint32_t i = 0;i < num_fea;i++){
       send[0] += (primal[i] - cons[i]) * (primal[i] - cons[i]);
       send[1] += (primal[i]) * (primal[i]);
       send[2] += (dual[i]) * (dual[i]);
     }

     MPI_Allreduce(send,recv,3,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);

     float prires  = sqrt(recv[0]);  /* sqrt(sum ||r_i||_2^2) */
     float nxstack = sqrt(recv[1]);  /* sqrt(sum ||x_i||_2^2) */
     float nystack = sqrt(recv[2]);  /* sqrt(sum ||y_i||_2^2) */
     
     float zdiff = 0.0;
     float z_squrednorm = 0.0; 

     for(uint32_t i = 0;i < num_fea;i++){
       zdiff += (cons[i] - cons_pre[i]) * (cons[i] - cons_pre[i]);
       z_squrednorm += cons[i] * cons[i];
     }

     float z_norm = sqrt(num_procs) * sqrt(z_squrednorm);
     float dualres = sqrt(num_procs) * rho * sqrt(zdiff); /* ||s^k||_2^2 = N rho^2 ||z - zprev||_2^2 */
     //double vmax = nxstack > z_norm?nxstack:z_norm;

     float eps_pri  = sqrt(num_procs * num_data)*ABSTOL + RELTOL * fmax(nxstack,z_norm);
     float eps_dual = sqrt(num_procs * num_data)*ABSTOL + RELTOL * nystack;

     if(rank == 0){
       char buf[500];
       snprintf(buf,500,"iter=%2d,prires=%.5f,eps_pri=%.5f,dualres=%.5f,eps_dual=%.5f",iter,prires,eps_pri,dualres,eps_dual);
       LOG(INFO) << buf;
       //LOG(INFO) << "iter=" << iter << ",prires=" << prires << ",eps_pri=" << eps_pri
       //          << ",dualres=" << dualres << ",eps_dual=" << eps_dual;
       //printf("%10.4f %10.4f %10.4f %10.4f\n", prires, eps_pri, dualres, eps_dual);
     }
     if(prires <= eps_pri && dualres <= eps_dual) {
       return true;
     }

     return false;
   }

  private:
   //training data
   std::string train_data;
   std::string pred_out;
   dmlc::RowBlockIter<unsigned> *dtrain;

   //admm parameter
   float *primal,*dual,*cons;
   float *w,*cons_pre;

   uint32_t num_fea;
   uint32_t num_data;
   int admm_max_iter;

   float rho;
   float l1_reg;
   float l2_reg;

   int rank; 
   int num_procs;

   //optimizer 
   std::string learner;
   Learner *optimizer;
   std::vector<Metric::pair_t> pair_vec;
   //configure
   std::map<std::string,std::string> cfg_;

   std::string model_in;
   std::string model_out;
};// class ADMM

}// namespace adPredictAlgo

#endif

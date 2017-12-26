#ifndef _ADPREDICTALGO_SOVLERSERVER_H_
#define _ADPREDICTALGO_SOVLERSERVER_H_

#include <iostream>
#include <mpi.h>
#include <map>
#include <vector>
#include <string>
#include <cassert>
#include <cstring>
#include <dmlc/data.h>

#include "metric.h"
#include "logisticreg.h"

namespace adPredictAlgo {

class SovlerServer {
  public:
    SovlerServer():n(NULL),z(NULL)
    {
      alpha = 0.01;
      beta = 1.;
      l1_reg = 0;
      l2_reg = 0.1;

      num_fea = 0;
    }

    virtual ~SovlerServer()
    {
      if(n != NULL)
        delete [] n;
      if(z != NULL)
        delete [] z;
    }

    inline void Init()
    {
      assert(num_fea != 0);
      if(!n)
        n = new float[num_fea];
      if(!z)
        z = new float[num_fea];

      memset(n,0.0,num_fea);
      memset(z,0.0,num_fea);

      LOG(INFO) << "1 SolverServer Start ," << num_procs - 1 << " SolverWorkers Start";

      dtest = dmlc::RowBlockIter<unsigned>::Create(
                          test_data.c_str(),
                          0,
                          1,
                          "libsvm"
                          );
      lr.Init();
    }

    inline void SetParam(const char *name,const char *val)
    {
      if(!strcmp(name,"alpha"))
        alpha = static_cast<float>(atof(val));
      if(!strcmp(name,"beta"))
        beta = static_cast<float>(atof(val));
      if(!strcmp(name,"l1_reg"))
        l1_reg = static_cast<float>(atof(val));
      if(!strcmp(name,"l2_reg"))
        l2_reg = static_cast<float>(atof(val));
      if(!strcmp(name,"num_fea"))
        num_fea = static_cast<uint32_t>(atoi(val));
      if(!strcmp(name,"num_procs"))
        num_procs = static_cast<int>(atoi(val));
      if(!strcmp(name,"num_epochs"))
        num_epochs = static_cast<int>(atoi(val));
      if(!strcmp(name,"test_data"))
        test_data = val;

      lr.SetParam(name,val);
    }

    void Start()
    {
      int iter = 0;
      while(iter < num_epochs){
        this->UpdateOneIter();
        iter++;
      }
      TaskPred();
      DumpModel();
    }

    void UpdateOneIter()
    {
      std::vector<uint32_t> tmp;
      std::vector<std::vector<uint32_t> > recv_keys;
      recv_keys.resize(num_procs - 1,tmp);

      std::vector<int> recv_keys_num(num_procs - 1);
      std::vector<std::vector<float> > grad;

      //recv the number of keys from work
      RecvKeyNumFromWork(recv_keys_num);
      //recv keys from work
      RecvKeysFromWork(recv_keys, recv_keys_num);
      //send the weight to every works
      SendWeightToWork(recv_keys);
      //Recv grad from every works
      RecvGradFromWork(recv_keys, grad);
      //update param
      UpdateModel(grad, recv_keys);

      //TaskPred();
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

    inline float Sigmoid(float inx)
    {
        return 1. / (1 + std::exp(-inx));
    }

    inline float PredIns(dmlc::Row<unsigned> v)
    {
        float sum = 0;
       for(unsigned i = 0;i < v.length;i++)
        {
            sum += lr.w[v.index[i]];
        }
        return Sigmoid(sum);
    }

    void UpdateModel(const std::vector<std::vector<float> > &grad,
                     const std::vector<std::vector<uint32_t> > &keys)
    {
      std::map<uint32_t,float> g;
      for(int i = 1; i < num_procs;i++)
      {
        for(size_t j = 0;j < keys[i-1].size();++j)
        {
          uint32_t fea_index = keys[i-1][j];
          if(g.count(fea_index))
            g[fea_index] = grad[i-1][j];
          else  
            g[fea_index] += grad[i-1][j];
        }
      }
      std::map<uint32_t,float>::iterator iter ;
      for(iter = g.begin() ; iter != g.end();iter++)
      {
        uint32_t fea_index = iter->first;
        if(fea_index >= num_fea)
        {
          std::cout << fea_index << " " << "out of the bound" <<std::endl;
        }

        float g_val = iter->second;
        float w_val = lr.w[fea_index];

        float theta = (std::sqrt(n[fea_index] + g_val * g_val) \
                            - std::sqrt(n[fea_index])) / alpha;
        z[fea_index] += g_val - theta * w_val;
        n[fea_index] += g_val * g_val;
        //update model
        if (fabs(z[fea_index]) < l1_reg) {
           lr.w[fea_index] = 0;
        }else{
           lr.w[fea_index] = (Sign(z[fea_index]) * l1_reg - z[fea_index]) / \
                (l2_reg + (beta + std::sqrt(n[fea_index])) / alpha);
        }        
      }
    }

    int Sign(float inx)
    {
      return inx > 0 ? 1:-1;
    }

    void RecvGradFromWork(const std::vector<std::vector<uint32_t> > &keys,
                          std::vector<std::vector<float> > & grad)
    {
      MPI_Status status;
      grad.clear();
      for(int i = 1;i < num_procs;++i)
      {
        std::vector<uint32_t> k = keys[i-1];
        std::vector<float> g;
        g.resize(k.size(),0.0f);

        MPI_Recv(&g[0], k.size(), MPI_FLOAT, i, 102, MPI_COMM_WORLD, &status);
        grad.push_back(g);
      }
    }

    void RecvKeyNumFromWork(std::vector<int> &recv_keys_num)
    {
      //recv_keys_num.clear();
      MPI_Status status;
      for(int i = 1;i < num_procs;++i)
      {
        MPI_Recv(&recv_keys_num[i-1],1, MPI_INT, i, 99, MPI_COMM_WORLD, &status);
      }
    }

    void RecvKeysFromWork(std::vector<std::vector<uint32_t> > &keys,
                          const std::vector<int> & key_num)
    {
      MPI_Status status;
      keys.clear();
      for(int i = 1;i < num_procs;++i)
      {
        std::vector<uint32_t> pre_procs_keys(key_num[i - 1]);
        MPI_Recv(&pre_procs_keys[0],key_num[i - 1], MPI_INT, i, 100, MPI_COMM_WORLD, &status);
        keys.push_back(pre_procs_keys);
      }
    }

    void SendWeightToWork(std::vector<std::vector<uint32_t> > &keys)
    {
      for(int i = 1;i < num_procs;++i)
      {
        std::vector<float> w;
        for(size_t j = 0;j < keys[i - 1].size();j++)
        {
          uint32_t fea_index = keys[i - 1][j];
          w.push_back(lr.w[fea_index]);
        }
        MPI_Send(&w[0],keys[i - 1].size(), MPI_FLOAT, i, 101, MPI_COMM_WORLD);
      }
    }

    virtual void DumpModel()
    {
      std::ofstream os("model.dat");
      lr.DumpModel(os);
      os.close();
    }
  private:
    float alpha,beta;
    float *n,*z;

    float l1_reg,l2_reg;

    int num_procs;
    int num_epochs;

    uint32_t num_fea;
    //FFMModel ffm;
    LRModel lr;
    std::string test_data;
    dmlc::RowBlockIter<unsigned> *dtest;
    std::vector<Metric::pair_t> pair_vec;
};
}

#endif

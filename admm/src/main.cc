//author:fangl@bayes.com
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <string>
#include <ctime>
#include <vector>

#include <dmlc/data.h>
#include <dmlc/io.h>
#include "config.h"
#include "admm.h"

namespace adPredictAlgo {

void TaskTrain(const std::vector<std::pair<std::string,std::string> > &cfg, 
               dmlc::RowBlockIter<unsigned> *dtrain)
{
  ADMM *admm = new ADMM(dtrain);
  admm->Configure(cfg);
  admm->Init();
  admm->TaskTrain();
  delete admm;
}

void TaskPred(const std::vector<std::pair<std::string,std::string> > &cfg,
              dmlc::RowBlockIter<unsigned> *dtrain)
{
  ADMM *admm = new ADMM(dtrain);
  admm->Configure(cfg);
  admm->Init();
  admm->LoadModel();
  admm->TaskPred();
  delete admm;
}

int RunTask(int argc,char **argv)
{
  if(argc < 2){
    LOG(FATAL) << "Usage:train conf_file";
    return 0;
  }
  // initialize MPI
  int rank, numprocs;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<std::pair<std::string,std::string> > cfg;
  adPredictAlgo::ConfigIterator itr(argv[1]);

  // store the rank and the number of processes
  cfg.push_back(std::make_pair(std::string("rank"), std::string(std::to_string(rank))));
  cfg.push_back(std::make_pair(std::string("num_procs"), std::string(std::to_string(numprocs))));

  // store conf file parameter
  while(itr.Next()) {
    cfg.push_back(std::make_pair(std::string(itr.name()),std::string(itr.val())));
  }

  // store comand line paramter
  char name[256],val[256];
  bool is_train = true;
  std::string train_data;
  for(int i = 2; i < argc;++i) {
    if(sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
      cfg.push_back(std::make_pair(std::string(name), std::string(val)));
      if(strcmp(name, "task") == 0 && strcmp(val, "pred") == 0) {
        is_train = false;
      }
      if(strcmp(name, "train_data") == 0) {
        train_data = std::string(val);
      }
    }
  }
  train_data += std::to_string(rank);
  dmlc::RowBlockIter<unsigned> *dtrain 
     = dmlc::RowBlockIter<unsigned>::Create
     (train_data.c_str(), 
      0, 
      1, 
     "libsvm");
  dtrain->BeforeFirst();
  uint32_t num_data = 0;
  size_t local_dim = dtrain->NumCol();
  while(dtrain->Next())  {
    const dmlc::RowBlock<unsigned> &batch = dtrain->Value();
    num_data += batch.size;
  }   
  size_t global_dim = 0, num_fea = 0;
  MPI_Allreduce(&local_dim, &global_dim,  1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); 
  num_fea = std::max(num_fea, global_dim) + 1;
  cfg.push_back(std::make_pair(std::string("num_fea"), std::to_string(num_fea)));
  cfg.push_back(std::make_pair(std::string("num_data"), std::to_string(num_data)));
  
  if(is_train) {
    TaskTrain(cfg, dtrain);
  } else {
    TaskPred(cfg, dtrain);
  }
  MPI_Finalize();
  return EXIT_SUCCESS;
}

} // namespace adPredictAlgo

int main(int argc,char **argv)
{
    return adPredictAlgo::RunTask(argc,argv);
}

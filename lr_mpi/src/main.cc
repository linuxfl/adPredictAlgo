//author:fangl@bayes.com
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <string>
#include <ctime>
#include <vector>

#include "ftrl.h"

namespace adPredictAlgo {

int RunTask(int argc,char **argv)
{
/*  if(argc < 2){
    std::cout << "Usage:train conf_file" << std::endl;
    return 0;
  }*/
  // initialize MPI
  int myid, numprocs;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  FTRLSovler *ftrl = new FTRLSovler();
  
  char rank[10] = {'0'},num_procs[100] = {'0'};
  sprintf(rank,"%d",myid);
  sprintf(num_procs,"%d",numprocs);
  ftrl->SetParam("rank",rank);
  ftrl->SetParam("num_procs",num_procs);

  // store comand line paramter
  char name[256],val[256];
  for(int i = 1; i < argc;++i) {
    if(sscanf(argv[i],"%[^=]=%s",name,val) == 2) {
      //std::cout << name << " " << val << std::endl;
      ftrl->SetParam(name,val);
    }
  }

  ftrl->Run();
//  if(ftrl)
//    delete ftrl;
  MPI_Finalize();
  return EXIT_SUCCESS;
}

} // namespace adPredictAlgo

int main(int argc,char **argv)
{
    return adPredictAlgo::RunTask(argc,argv);
}

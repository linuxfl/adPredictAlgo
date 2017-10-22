#include <iostream>
#include <dmlc/data.h>
#include <dmlc/io.h>

#include "lbfgs.h"

int main(int argc,char **argv) 
{
  if(argc < 2) {
    std::cerr << "Usage:train dtrain param=val";
  }

  dmlc::RowBlockIter<unsigned> *dtrain 
    = dmlc::RowBlockIter<unsigned>::Create
    (argv[1],
      0,
      1,
    "libsvm");

/*  dmlc::RowBlockIter<unsigned> *dtest
    = dmlc::RowBlockIter<unsigned>::Create
    (argv[2],
      0,
      1,
    "libsvm");
  */
  adPredictAlgo::LBFGSSolver *lbfgs
    = new adPredictAlgo::LBFGSSolver(dtrain);
  
  char name[256],val[256];
  for(int i = 2;i < argc;i++) {
    sscanf(argv[i],"%[^=]=%s",name,val);
    lbfgs->SetParam(name,val);
  }
  lbfgs->Run();
  delete lbfgs;

  return 0;
}

#include "../include/ftrl_train.h"

int main(int argc,char **argv)
{
  if(argc < 3)
  {
    LOG(FATAL) << "Usage:train_data param=val";
    return 0;
  }

  dmlc::RowBlockIter<unsigned> *test_data
      = dmlc::RowBlockIter<unsigned>::Create
      (argv[2],
      0,  
      1,  
      "libsvm"
      );  
 
  adPredictAlgo::FTRL *ftrl = new adPredictAlgo::FTRL(argv[1],test_data);
  
  char name[256],val[256];
  for(int i = 2;i < argc;i++)
  {
    sscanf(argv[i],"%[^=]=%s",name,val);
    ftrl->SetParam(name,val);
  }
  ftrl->Run();
  if(ftrl != nullptr)
    delete ftrl;
  return 0;
}

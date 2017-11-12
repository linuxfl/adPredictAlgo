#include "../include/ftrl_train.h"

int main(int argc,char **argv)
{
  if(argc < 3)
  {
    LOG(FATAL) << "Usage:train_data param=val";
    //std::cerr << "Usage:train_data test_data param=val" << std::endl;
    return 0;
  }

  adPredictAlgo::FTRL *ftrl = nullptr;
  
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

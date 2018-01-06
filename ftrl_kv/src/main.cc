#include <iostream>
#include "ftrl.h"

int main(int argc,char **argv)
{
  if(argc < 2)
  {
    std::cout << "Usage:train train_data param=val" << std::endl;
    return 1;
  }
  
  algo::FTRLSolver *ftrl = new algo::FTRLSolver(argv[1]);
  
  char name[128],val[128];
  for(int i = 2;i < argc;++i)
  {
    if(sscanf(argv[i],"%[^=]=%s",name,val) == 2){
      ftrl->SetParam(name,val);
    }
  }

  ftrl->Run();

  if(ftrl)
    delete ftrl;
  return 0;
}

#include "../include/ftrl.h"

int main(int argc,char **argv)
{
  if(argc < 4)
  {
    LOG(FATAL) << "Usage:train_data test_data memory_in param=val";
    //std::cerr << "Usage:train_data test_data param=val" << std::endl;
    return 0;
  }

  dmlc::RowBlockIter<unsigned> *test_data
      = dmlc::RowBlockIter<unsigned>::Create
      (argv[2],
      0,
      1,
      "libsvm"
      );

  algo::Ftrl *ftrl = nullptr;
  
  if (!strcmp(argv[3],"batch")) {
    dmlc::RowBlockIter<unsigned> *train_data
        = dmlc::RowBlockIter<unsigned>::Create
        (argv[1],
        0,
        1,
        "libsvm"
        );
    ftrl = new algo::Ftrl(train_data,test_data);
  }else if (!strcmp(argv[3],"stream")) {
    ftrl = new algo::Ftrl(argv[1],test_data);
  }else {
    LOG(FATAL) << "error memory_in type!";
  }
  
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

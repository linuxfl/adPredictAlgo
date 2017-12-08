#include <string>
#include "../include/ftrl.h"

std::string train_help()
{
  return std::string(
  "usage: adpa_ftrl training_set_file test_set_file param=val\n"
  "\n"
  "param:\n"
    "*num_fea: the number of the feature (default 0)\n"
    "l1_reg: l1 regularization co-efficient (default 0.)\n"
    "l2_reg: l2 regularization co-efficient (default 0.1)\n"
    "alpha: ftrl parameter,please refer in relative paper (default 0.01)\n"
    "beta: ftrl parameter,please refer in relative paper (default 1)\n"
    "model_in: input model file,when model_in is not \"NULL\",launch online model\n"
    "model_out: output model file\n");
}

int main(int argc,char **argv)
{
  if(!strcmp(argv[1],"-h") || !strcmp(argv[1],"-help")) {
    std::cout << train_help() << std::endl;
    return 0;
  }

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

  adPredictAlgo::Ftrl *ftrl = nullptr;
  
  if (!strcmp(argv[3],"batch")) {
    dmlc::RowBlockIter<unsigned> *train_data
        = dmlc::RowBlockIter<unsigned>::Create
        (argv[1],
        0,
        1,
        "libsvm"
        );
    ftrl = new adPredictAlgo::Ftrl(train_data,test_data);
  }else if (!strcmp(argv[3],"stream")) {
    ftrl = new adPredictAlgo::Ftrl(argv[1],test_data);
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

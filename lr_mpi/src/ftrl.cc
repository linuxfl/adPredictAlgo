#include <iostream>

#include "ftrl.h"

namespace adPredictAlgo {

FTRLSovler::FTRLSovler()
{
}

FTRLSovler::~FTRLSovler()
{
}

void FTRLSovler::SetParam(const char *name,const char *val)
{
  if(!strcmp(name,"rank"))
    rank = static_cast<int>(atoi(val));
  
  ss.SetParam(name,val);
  sw.SetParam(name,val);
}

void FTRLSovler::Run()
{
  if(rank == 0){
    ss.Init();
    Timer t;
    t.Start();
    ss.Start();
    t.Stop();
    LOG(INFO) << "Elapsed time:" << t.ElapsedSeconds() << " sec.";
  }else{
    sw.Init();
    sw.Start();
  }
}

}

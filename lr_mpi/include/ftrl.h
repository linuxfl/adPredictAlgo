#ifndef _ADPREDICT_FTRL_H_
#define _ADPREDICT_FTRL_H_

#include <iostream>
#include "SovlerServer.h"
#include "SovlerWorker.h"
#include "elapse.h"

namespace adPredictAlgo {

class FTRLSovler {
  public:
    FTRLSovler();
    virtual ~FTRLSovler();

    virtual void Run();
    virtual void SetParam(const char *name,const char *val);

  private:
    int rank;
    //server
    SovlerServer ss;
    //worker
    SovlerWorker sw;
};

}
#endif

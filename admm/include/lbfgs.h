#include "learner.h"

namespace adPredictAlgo {

class LBFGSSolver : public Learner {
  public:
    LBFGSSolver() {
      
    }

    ~LBFGSSolver() {
      
    }

    void Configure(const std::vector<std::pair<std::string,std::string> > &cfg) 
    {
      std::cout << "lbfgs configure" << std::endl;
    }
    
    float PredIns(const dmlc::Row<unsigned> &v,
                  const float *w){
      return 0.0;
    }
    void Train(float *primal,
           float *dual,
           float *cons,
           float rho,
           dmlc::RowBlockIter<unsigned> *dtrain)
    {
      std::cout << "lbfgs train" << std::endl;
    }
};

}

#include "learner.h"

namespace adPredictAlgo {

class LBFGSSolver : public Learner {
  public:
    LBFGSSolver() {
      linesearch_c1 = 1e-4f;
      linesearch_backoff = 0.5f;
      max_linesearch_iter = 20;
      lbfgs_stop_tol = 1e-5f;
      max_lbfgs_iter = 10;
      memory_size = 4;
      num_fea = 0;
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
      int iter = 0;
      while(iter < max_lbfgs_iter) {
        if(this->UpdateOneIter(primal,dual,cons,rho,iter,dtrain))
          break;
        iter++;
      }
    }

  private:
    float l1_reg;
    float linesearch_c1;
    float linesearch_backoff;
    int max_linesearch_iter;
    float lbfgs_stop_tol;
    int max_lbfgs_iter;
    size_t memory_size;
    size_t num_fea;
    
    double old_objval;
    double new_objval;
    double init_objval;

    std::vector<float> alpha;
    std::vector<float *> y;
    std::vector<float *> s;

    float *z;
    float *grad;
    float *old_weight;

    void Init() {
      float *e = new float[num_fea];
      memset(e,0.0,sizeof(float) * num_fea);
      y.resize(memory_size,e);
      s.resize(memory_size,e);

      alpha.resize(memory_size,0.0);
      
    }
    //one update
    bool UpdateOneIter(float *prima,
                       float *dual,
                       float *cons,
                       float rho,
                       int iter,
                       dmlc::RowBlockIter<unsigned> *dtrain)
    {
      bool stop = false;
      float vdot = FindChangeDirection();
      BacktrackLineSearch();
      UpdateHistInfo();
      if(old_objval - new_objval < lbfgs_stop_tol * init_objval)
        return true;
      return stop;
    }

    float FindChangeDirection()
    {

    }

    void BacktrackLineSearch()
    {

    }

    void UpdateHistInfo()
    {

    }
};

}

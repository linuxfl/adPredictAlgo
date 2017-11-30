#include "learner.h"

namespace adPredictAlgo {

class LBFGSSolver : public Learner {
    public:
        LBFGSSolver() {
					
        }

        ~LBFGSSolver() {
					
        }

        void Configure(std::vector<std::pair<std::string,std::string> > cfg) 
				{

				}

        void Train(float *primal,
									 float *dual,
								   float *cons,
                   float rho,
									 dmlc::RowBlockIter<unsigned> *dtrain)
				{
					
				}
};

}

#include "learner.h"

namespace adPredictAlgo {

class FTRL : public Learner {
    public:
        FTRL() {
					
        }

        ~FTRL() {
					
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

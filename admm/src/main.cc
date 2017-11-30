//author:fangl@bayes.com
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <string>
#include <ctime>

#include <dmlc/data.h>
#include <dmlc/io.h>
#include "config.h"

namespace adPredictAlgo {

enum Task {
    kTrain = 0,
    kPredict = 1
};

struct ADMMParam : public dmlc::Parameter<ADMMParam> {

    inline void Configure(std::vector<std::pair<std::string,std::string> >){

    }
};

void TaskTrain(const ADMMParam &param)
{
    ADMM admm(param);
    admm.TaskTrain();
}

void TaskPred(const ADMMParam &param)
{
    ADMM admm(param);
    admm.TaskPred();
}

int RunTask(int argc,char **argv)
{
	if(argc < 2){
	    LOG(FATAL) << "Usage:train conf_file" << endl;
		return 0;
	}
	
	// initialize MPI
	int myid, numprocs;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    
    std::vector<std::pair<std::string,std::string> > cfg;
    adPredictAlgo::ConfigIterator itr(argv[1]);

    while(itr.next()) {
        cfg.push_back(std::make_pair(std::string(itr.name()),std::string(itr.val())));
    }

    for(int i = 2; i < argc;++i) {
        char name[256],val[256];
        if(sscanf(argv[i],"%[^=]=%s",name,val) == 2) {
            cfg.push_back(std::make_pair(std::string(name),std::string(val)));
        }
    }

    ADMMParam param;
    param.Configure(cfg);

    switch(param.task) {
        case kTrain: TaskTrain(param); break;
        case kPredict: TaskPredict(param);break;
    }
    
    MPI_Finalize();
	return EXIT_SUCCESS;
}

} // namespace adPredictAlgo

int main(int argc,char **argv)
{
    return RunTask(argc,argv);
}

//author:fangl@bayes.com
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <string>
#include <ctime>
#include <vector>
#include <pair>

#include <dmlc/data.h>
#include <dmlc/io.h>
#include "config.h"
#include "admm.h"

namespace adPredictAlgo {

enum Task {
    kTrain = 0,
    kPredict = 1
};

struct AppParam : public dmlc::Parameter<AppParam> {
    //the task name
    std::string task;
    //whether silent
    int slient;
    //all the configurations
    std::vector<std::pair<std::string,std::string> > cfg;

    DMLC_DECLARE_PARAMETER(AppParam){
        DMLC_DECLARE_FIELD(task).set_default(kTrain)
            .add_enum("train",kTrain)
            .add_enum("pred",kPred)
        DMLC_DECLARE_FIELD(slient).set_default(1).set_range(0,2);
    }
    inline void Configure(std::vector<std::pair<std::string,std::string> > cfg){
        this->cfg = cfg;
        this->InitAllowUnknow(cfg);
    }

};

DMLC_REGITER_PARAMETER(AppParam);

void TaskTrain(const AppParam &param)
{
    ADMM admm(param.cfg);
    admm.Configure();
    admm.Init();
    admm.TaskTrain();
}

void TaskPred(const AppParam &param)
{
    ADMM admm(param.cfg);
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

    // store the rank and the number of processes
    cfg.push_back(std::make_pair(std::string("rank"),std::string(myid)));
    cfg.push_back(std::make_pair(std::string("num_procs"),std::string(numprocs)));

    // store conf file parameter
    while(itr.next()) {
        cfg.push_back(std::make_pair(std::string(itr.name()),std::string(itr.val())));
    }

    // store comand line paramter
    for(int i = 2; i < argc;++i) {
        char name[256],val[256];
        if(sscanf(argv[i],"%[^=]=%s",name,val) == 2) {
            cfg.push_back(std::make_pair(std::string(name),std::string(val)));
        }
    }

    // init Application Pramamter
    AppParam param;
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

#include "worker.h"
#include "server.h"

#include "config.h"
#include "ps.h"

int main(int argc,char *argv[]){
  if(argc < 2)
  {
    LOG(FATAL) << "Usage: lr_ps conf_file";
  }
  std::vector<std::pair<std::string, std::string>> cfg;
  adPredictAlgo::ConfigIterator itr(argv[1]);
  while(itr.Next()) {
    cfg.push_back(std::make_pair(std::string(itr.name()),std::string(itr.val())));
  }

  if (ps::IsServer()) {
    //std::shared_ptr<adPredictAlgo::S> server = std::make_shared<adPredictAlgo::S>();
    auto server = std::shared_ptr<adPredictAlgo::S>();
    server->StartServer(cfg);
  }
  ps::Start();
  if (ps::IsWorker()) {
    adPredictAlgo::W* worker = new adPredictAlgo::W();
    //auto worker = std::shared_ptr<adPredictAlgo::W>();
    worker->Run(cfg);
  }
  ps::Finalize();
}

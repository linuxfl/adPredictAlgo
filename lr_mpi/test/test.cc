#include <iostream>
#include "../include/dataloader.h"
#include "/home/work/fangling/adPredictalgo/common/include/elapse.h"

int main(int argc,char **argv)
{
  adPredictAlgo::DataLoader data_loader;
  std::vector<std::pair<std::string,std::string> > cfg;
  cfg.push_back(std::make_pair("minibatch_size","1000"));
  cfg.push_back(std::make_pair("train_data","shitu"));
  cfg.push_back(std::make_pair("rank","1"));
  data_loader.Configure(cfg);

  adPredictAlgo::DataVec data;
  adPredictAlgo::Timer t;

  t.Start();
  data_loader.LoadAllDataFromFile(data);
//  data_loader.LoadData(data);
  t.Stop();
  std::cout << t.ElapsedSeconds()<< std::endl;
    for(size_t i = 0;i < data.size();i++)
  {
    std::cout << data[i].label << " ";
    for(size_t j = 0; j < data[i].fea_vec.size();j++)
      if(j != data[i].fea_vec.size() - 1){
        std::cout << data[i].fea_vec[j] << " ";
      }else{
        std::cout << data[i].fea_vec[j];
      }
    std::cout << std::endl;
  }

//  std::cout << data.size() << std::endl;
  return 0;
}

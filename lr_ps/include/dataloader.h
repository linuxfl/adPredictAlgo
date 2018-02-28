#ifndef _ADPREDICTALGO_IO_H_
#define _ADPREDICTALGO_IO_H_

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <cstdio>
#include <string>
#include <map>
#include <cassert>
#include <sstream>

#include "data.h"

namespace adPredictAlgo {

class DataLoader {
  public:
    DataLoader() {
      train_data = "NULL";
      rank = 0;
    }

    ~DataLoader() {
    }

    void Init() {
      assert(train_data != "NULL");
      train_data += std::to_string(rank);
    }
    
    void Configure(std::unordered_map<std::string, std::string> &cfg)
    {
      if(cfg.count("train_data"))
        train_data = cfg["train_data"];
      if(cfg.count("rank"))
        rank = static_cast<int>(atoi(cfg["rank"].c_str()));
      this->Init();
    }

    inline void Split(std::string & line,
                 std::vector<std::string> & fea_vec)
    {
      std::stringstream ss;
      ss << line;
      while(ss >> line){
        fea_vec.push_back(line);
      }
    }

    void LoadAllDataFromFile(DataVec & data)
    {
      std::ifstream is(train_data.c_str());
      std::string line;
      std::vector<std::string> fea_vec;
      while(getline(is, line)){
        fea_vec.clear();
        Split(line, fea_vec);
        Instance ins;
        ins.label = static_cast<int>(atoi(fea_vec[0].c_str()));
        for(unsigned int idx = 1;idx < fea_vec.size();++idx)
        {
          uint32_t fid = static_cast<KeyType>(atoi(fea_vec[idx].c_str()));
          ins.fea_vec.push_back(fid);
        }

        data.push_back(ins);
      }
    }

  private:
    //train data
    std::string train_data;
    //conf
    std::map<std::string,std::string> cfg_;
    //rank
    int rank;
};

}
#endif

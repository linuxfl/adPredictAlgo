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

#include "data.h"
#include "str_util.h"

namespace adPredictAlgo {

class DataLoader {
  public:
    DataLoader() {
      fp = NULL;
      buf = NULL;

      minibatch_size = 1;
      train_data = "NULL";
    }

    ~DataLoader() {
      delete buf;
      fclose(fp);
    }

    void Init() {
      assert(train_data != "NULL");

      if((fp = fopen(train_data.c_str(),"r")) == NULL){
        std::cout << "open train data error!" << std::endl;
        exit(1);
      }

      if(minibatch_size < 1)
        minibatch_size = 2;
      buff_size = 2048 * minibatch_size;

      buf = new char[buff_size];
      memset(buf,0,buff_size);
    }

    void Configure(const std::vector<std::pair<std::string,std::string> > &cfg)
    {
      for(const auto & kv : cfg)
      {
        cfg_[kv.first] = kv.second;
      }

      if(cfg_.count("train_data"))
        train_data = cfg_["train_data"];
      if(cfg_.count("minibatch_size"))
        minibatch_size = static_cast<unsigned int>(atoi(cfg_["minibatch_size"].c_str()));

      this->Init();
    }

    void LoadData(DataVec &data)
    {
      std::ifstream is(train_data.c_str());
      std::string line;
      Instance ins;
      while(getline(is,line))
      {
        ins.clear();
        std::vector<std::string> fea_vec;
        util::str_util::split(line," ",fea_vec);
        ins.label = static_cast<int>(atoi(fea_vec[0].c_str()));

        for(size_t i = 1;i < fea_vec.size();i++) {
          std::vector<std::string> kvs;
          ffm_node _node;
          util::str_util::split(fea_vec[i],":",kvs);

          _node.field_index = static_cast<uint32_t>(atoi(kvs[0].c_str()));
          _node.fea_index = static_cast<uint32_t>(atoi(kvs[1].c_str()));

          ins.fea_vec.push_back(_node);
        }   
        data.push_back(ins);
      }
    }

    void ReadBuf()
    {
      size_t read_buff_size = buff_size - 2048;
      size_t ret = fread(buf,sizeof(char),read_buff_size,fp);
      
      //may be not a line
      if(buf[ret - 1] != '\n' && !feof(fp))
      {
        char *p = &buf[ret];
        fgets(p,2048,fp);
      }

    }

    size_t ParseToData(DataVec &data)
    {
      char c[2048];
      char *p = buf;
      int k;
      size_t cnt = 0;
      Instance ins_;
      while(*p != '\0'){
        k = 0;
        //只有一行时可能没有换行
        while(*p != '\n' && *p != '\0'){
          c[k++] = *p++;
        }
        c[k] = '\0';
        p++;
        ins_.clear();
        ParseLine(c,ins_);
        data.push_back(ins_);
        cnt++;
      }
      return cnt;
    }

    bool LoadDataFromFile(DataVec &data)
    {
      assert(fp != NULL || buf != NULL);
      
      size_t read_line = 0;
      //while(!feof(fp)){
      memset(buf,0,buff_size);

      ReadBuf();
      read_line += ParseToData(data);
      //}
      return !feof(fp);
    }

  private:
    //train data
    std::string train_data;
    //file pointer
    FILE *fp;
    //size of minibatch
    unsigned int minibatch_size;
    //buffer store
    char *buf;
    //buffer size
    size_t buff_size;
    //conf
    std::map<std::string,std::string> cfg_;
    
  private:
    inline void ParseLine(char *buf_,Instance &ins)
    {
      ins.clear();
      ffm_node node_;
      char *p = buf_;
      char *q;
      uint32_t field_index;
      uint32_t fea_index;

      q = p;
      //parse label
      while(*q != ' ')
        q++;
      *q = '\0';
      ins.label = static_cast<int>(std::atoi(p));
      
      q++;
      p = q;
      
      //parse fea
      char c[128];
      while(*p != '\0')
      {
        int k = 0;
        while(*q != ' ' && *q != ':' && *q != '\0'){
          c[k++] = *q++;
        }

        c[k] = '\0';
        
        //store field index
        if(*q == ':')
          field_index = static_cast<uint32_t>(std::atoi(c));

        //end of the ffm kvs,store to ffm node
        if(*q == ' ' || *q == '\0'){
          fea_index = static_cast<uint32_t>(std::atoi(c));
        
          node_.field_index = field_index;
          node_.fea_index = fea_index;
          ins.fea_vec.push_back(node_);
        }

        if(*q != '\0')
          q++;
        
        p = q;
      }
    }

};

}
#endif

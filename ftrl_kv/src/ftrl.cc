#include <iostream>
#include "ftrl.h"
#include <cstring>

namespace algo {

void FTRLSolver::SetParam(const char *name, const char *val)
{
  if (!strcmp(name,"model_in")) 
    model_in = val;
  if (!strcmp(name,"model_out")) 
    model_out = val;
  if (!strcmp(name,"l1_reg")) 
    l1_reg = static_cast<float>(atof(val));
  if (!strcmp(name,"l2_reg")) 
    l2_reg = static_cast<float>(atof(val));
  if (!strcmp(name,"alpha")) 
    alpha = static_cast<float>(atof(val));
  if (!strcmp(name,"beta")) 
    beta = static_cast<float>(atof(val));
  if (!strcmp(name,"pred_out"))
    pred_out = val;
  if (!strcmp(name,"task"))
    task = val;
  if (!strcmp(name,"is_incre"))
    is_incre = static_cast<int>(atoi(val));
  if (!strcmp(name,"save_aux"))
    save_aux = static_cast<int>(atoi(val));
}

void FTRLSolver::TaskTrain()
{
  std::ifstream is(train_data);
  assert(is.fail() == false);

  std::string line;
  fea::instance ins;
  int cnt = 0;
  while(getline(is,line))
  {
    ins.reset();
    ParseLine(line, ins);
    TrainIns(ins);
    cnt++;
    if(cnt % 100000 == 0)
      std::cout << "train instance:" << cnt << " ,nnw:" << model.size() << std::endl;
  }
  is.close();
}

void FTRLSolver::TaskPred()
{
  std::ifstream is(train_data);
  assert(is.fail() == false);
  std::vector<std::pair<ValueType, int>> result;

  std::string line;
  fea::instance ins;
  int cnt = 0;
  while(getline(is,line)){
    ins.reset();
    ParseLine(line, ins);
    float fv = PredIns(ins);
    result.push_back(std::make_pair(fv,ins.label));
    ++cnt;
    if(cnt % 100000 == 0)
      std::cout << "predict instance : " << cnt << std::endl;
  }

  std::ofstream os(pred_out.c_str());
  assert(os.fail() == false);
  for(size_t i = 0;i < result.size();++i)
  {
    os << result[i].first << " " << result[i].second << std::endl;
  }

  os.close();
  is.close();
}

void FTRLSolver::TrainIns(const fea::instance &ins)
{
  std::vector<uint32_t> fea_vec = ins.fea_vec;
  size_t ins_len = fea_vec.size();
  std::vector<ftrlentry> e_vec;
  for(size_t idx = 0;idx < ins_len;++idx)
  {
    uint32_t fid = fea_vec[idx];
    ftrlentry e;
    if(model.count(fid))
      e = model[fid];
    
    if (fabs(e.z) < l1_reg)
    {
      e.w = 0;
    }else{
      e.w = (Sign(e.z) * l1_reg - e.z) \
              / (l2_reg + (beta + sqrt(e.n)) / alpha);
    }
    e_vec.push_back(e);
  }

  //calc grad
  ValueType g = PredIns(ins) - ins.label;

  for(size_t idx = 0;idx < ins_len;++idx)
  {
    uint32_t fid = fea_vec[idx];
    ftrlentry &e = e_vec[idx];
    ValueType theta = (sqrt(e.n + g * g) - sqrt(e.n)) / alpha;
    e.z += g - theta * e.w;
    e.n += g * g;
    model[fid] = e;
  }
}

void FTRLSolver::SaveModel() const {
  std::ofstream os(model_out.c_str());
  assert(os.fail() == false);

  for(const auto &it : model)
  {
    if(save_aux){
      os << it.first << " " 
         << (it.second).w << " " 
         << (it.second).z << " " 
         << (it.second).n << std::endl;
    }else{
      os << it.first << " "
         << (it.second).w << std::endl;
    }
  }
  os.close();
}

void FTRLSolver::LoadModel() {
  std::ifstream is(model_in.c_str());
  assert(is.fail() == false);
  std::cout << "Load Model now..." << std::endl;

  uint32_t fid;
  ftrlentry e;
  if(save_aux){
    while(!is.eof()){
      is >> fid >> e.w >> e.z >> e.n;
      model[fid] = e;
    }
  }else{
    while(!is.eof()){
      is >> fid >> e.w;
      if(e.w != 0.0)
        model[fid] = e;
    }
  } 

  is.close();
}

}

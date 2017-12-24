#ifndef _ADPREDCTALGO_DATA_
#define _ADPREDCTALGO_DATA_

#include <iostream>
#include <vector>

namespace adPredictAlgo {

typedef struct ins {
  int label;
  std::vector<uint32_t> fea_vec;
  ins(){
    //do nothing
  }

  ~ins(){
    fea_vec.clear();
  }
  void clear()
  {
    label = 0;
    fea_vec.clear();
  }
}Instance;

typedef std::vector<Instance> DataVec;

}
#endif

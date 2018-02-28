#ifndef _ADPREDCTALGO_DATA_
#define _ADPREDCTALGO_DATA_

#include <iostream>
#include <vector>
#include "ps.h"

namespace adPredictAlgo {

typedef float ValueType;
typedef ps::Key KeyType;

typedef struct ins {
  int label;
  std::vector<KeyType> fea_vec;
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

Factorization Machines
====
* input format:libSVM
* run command:./bin/adpa_fm dtrain dtest param=val
* example:./bin/adpa_fm dtrain dtest num_fea=100000 ffm_dim=3

Parameters
====
All the parameters can be set by param=value

#### Important Parameters
* l1_reg [default = 0]
  - l1 regularization co-efficient
* l2_reg [default = 0.1]
  - l2 regularization co-efficient
* l1_fm_reg [default = 0]
  - l1 regularization co-efficient for ffm model
* l2_fm_reg [default = 0.1]
  - l2 regularization co-efficient for ffm model
* alpha [default = 0.1]
  - ftrl parameter,please refer in relative paper
* beta [default = 1]
  - ftrl parameter,please refer in relative paper
* alpha_fm [default = 0.1]
  - ftrl parameter,please refer in relative paper
* beta_ffm [default = 0.1]
  - ftrl parameter,please refer in relative paper
* model_in [default = "NULL"]
  - input model file
* model_out [default = "ffm_model.dat"]
  - output model file
* num_fea [default = 0]
  - number of feature
* fm_dim [default = 0]
  - dim of fm model

Ftrl for Logistic Regression
====
* input format: LibSVM
* run command:./bin/train dtrain dtest memory_in param=val
* memory_in: batch or stream

Parameters
====
All the parameters can be set by param=value

#### Important Parameters
* l1_reg [default = 0]
  - l1 regularization co-efficient
* l2_reg [default = 0.1]
  - l2 regularization co-efficient
* alpha [default = 0.01]
  - ftrl parameter,please refer in relative paper
* beta [default = 1]
  - ftrl parameter,please refer in relative paper
* model_in [default = "NULL"]
  - input model file,when model_in is not "NULL",launch online model
* model_out [default = "lr_model.dat"]
  - output model file
* num_feature [default = 0]
  - feature dim

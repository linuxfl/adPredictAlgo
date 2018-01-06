Ftrl for Logistic Regression(key-values)
====
* input format: LibSVM
* run command:./bin/train dtrain dtest param=val

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
* save_aux [default = 1]
  - whether save the ftrl paramter n and z
* is_incre [default = 0]
  - whether online learning

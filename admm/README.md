Alternating Direction Method of Multipliers for Logistic Regression
====
* input format: LibSVM
* run command:mpirun -np 2 ./bin/admm ./conf/xxx.conf param=val

Parameters
====
All the parameters can be set by param=value

#### Important Parameters
* l1_reg [default = 0]
  - l1 regularization co-efficient
* l2_reg [default = 0.1]
  - l2 regularization co-efficient
* admm_max_iter [default = 5]
  - maximum number of admm iterations
* model_in [default = "NULL"]
  - input model file,when model_in is not "NULL",launch online model
* model_out [default = "lr_model.dat"]
  - output model file
* num_fea [default = 0]
  - feature dim

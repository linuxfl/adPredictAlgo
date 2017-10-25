LBFGS for Linear and Logistic Regression
====
* input format: LibSVM
* run command:./bin/lbfgs dtrain param=val

Parameters
====
All the parameters can be set by param=value

#### Important Parameters
* l1_reg [default = 0]
  - l1 regularization co-efficient
* l2_reg [default = 0.1]
  - l2 regularization co-efficient
* model_in [default = "NULL"]
  - input model file
* model_out [default = "lr_model.dat"]
  - output model file
* num_fea [default = 0]
  - feature dim
* task [default = train]
  - train and pred mode
* lbfgs_stop_tol [default = 1e-4]
  - relative tolerance level of loss reduction with respect to initial loss
* max_lbfgs_iter [default = 100]
  - maximum number of lbfgs iterations

### Optimization Related parameters
* min_lbfgs_iter [default = 5]
  - minimum number of lbfgs iterations
* max_linesearch_iter [default = 100]
  - maximum number of iterations in linesearch
* linesearch_c1 [default = 1e-4]
  - c1 co-efficient in backoff linesearch
* linesarch_backoff [default = 0.5]
  - backoff ratio in linesearch

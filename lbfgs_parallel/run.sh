dmlc-core/tracker/dmlc-submit --cluster mpi --num-workers 1 ./lbfgs.dmlc shitu_train_origin reg_L2=0.2 max_lbfgs_iter=200
./lbfgs.dmlc shitu_test_origin task=pred model_in=final.model 

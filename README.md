# Kaggle_HomeDepot
Turing Test's Solution for Home Depot Product Search Relevance Competition on Kaggle (https://www.kaggle.com/c/home-depot-product-search-relevance)

## Submission
| Submission | CV RMSE | Public LB RMSE | Private LB RMSE | Position |
| :---: | :-------: | :--------------: | :---------------: | :------------------: |
| [Best Single Model from Chenglong](./Output/Subm/test.pred.[Feat@basic_nonlinear_201604210409]_[Learner@reg_xgb_tree]_[Id@84].[Mean0.438318]_[Std0.000786].csv) | 0.43832 | 0.43996 | 0.43811 | 9 |
| [Best Ensemble Model from Igor and Kostia](./Output/Subm/submission_kostia + igor final_ensemble (1 to 3 weights).csv) | - | 0.43819 | 0.43704 | 8 |
| [Best Ensemble Model from Chenglong](./Output/Subm/test.pred.[Feat@level2_meta_linear_201605030922]_[Learner@reg_ensemble]_[Id@1].[Mean0.436087]_[Std0.001027].csv) | 0.43609 | 0.43582 | 0.43325 | 4 |
| [Best Final Ensemble Model](./Output/Subm/reproduced_blend_0.438_0.436CV.csv) | - | 0.43465 | 0.43248 | 3 |

## FlowChart

## Documentation

See `./Doc/Kaggle_HomeDepot_Turing_Test.pdf` for documentation.

## Instruction

### Chenglong's Part
Before proceeding, you should have placed all the data from the [competition website](https://www.kaggle.com/c/home-depot-product-search-relevance/data) into folder `./Data`. 

Note that in the following, all the commands and scripts are excuted and run in directory `./Code/Chenglong`.

#### Step 1. Generate Features
To generate data and features, you can run `python run_data.py`. While we have tried our best to make things as parallelism and efficient as possible, this part might still take 1 ~ 2 days to finish, depending on your computational power. So be patient :)

#### Step 2. Generate Feature Matrix
In step 1, we have generated a few thousands of feaures. However, only part of them will be used to build our model. For example, we don't need those features that have very little predictive power (e.g., have very small correlation with the target relevance.) Thus we need to do some feature selection.

In our solution, feature selection is enabled via the following two successive steps.
#####1. `regex` Style Manual Feature Selection
The general idea is to include or exclude specific feautres via `regex` operations of the feature names. For example, 
- one can specify the features that he want to include via the `MANDATORY_FEATS` variable, desipte of its correlation with the target 
- one can also specify the features that he want to exclude via the `COMMENT_OUT_FEATS` variable, desipte of its correlation with the target (`MANDATORY_FEATS` has higher priority than `COMMENT_OUT_FEATS`.)

This approach is implemented as `get_feature_conf_*.py`. The output of this is a feature conf file. For example, after running the following command:
`python get_feature_conf_nonlinear.py -d 10 -o feature_conf_nonlinear_201605010058.py`
we will get a new feature conf `./conf/feature_conf_nonlinear_201605010058.py` which contains a feature dictionary specifying the features to be included in the following step.

One can play around with `MANDATORY_FEATS` and `COMMENT_OUT_FEATS` to generate different feature subset. We have included in `./conf` a few other feature confs from our final submission. Among them, `feature_conf_nonlinear_201604210409.py` is used for the best single model.

#####2. Correlation based Feature Selection
With the above generated feature conf, you can combine all the features into a feature matrix via the following command:
`python feature_combiner.py -l 1 -c feature_conf_nonlinear_201604210409 -n basic_nonlinear_201604210409 -t 0.05`

The `-t 0.05` above is used to enable the correlation base feature selection. In this case, it means: drop any feature that has a correlation coef lower than `0.05` with the target relevance.

#### Step 3. Generate Submission
#####1. Various Tasks
In our solution, a `task` is an object composite of a specific `feature` (e.g., `basic_nonlinear_201604210409`) and a specific `learner` (`XGBoostRegressor` from [xgboost](https://github.com/dmlc/xgboost)). The definitions for `task`, `feature` and `learner` are in `task.py`.

During the competition, we have run various tasks to generate a diverse 1st level model library. Please see `./Log/level1_models` for all the tasks we have included in our final submission.

#####2. Best Single Model
After generating the `feature` `basic_nonlinear_201604210409` (see step 2 how to generate this), run the following command to generate the best single model:
`python task.py -m single -f basic_nonlinear_201604210409 -l reg_xgb_tree_best_single_model -e 1`

This should generate a submission with local CV RMSE around 0.438 ~ 0.439. (The best single model we have generated is [here](./Output/Subm/test.pred.[Feat@basic_nonlinear_201604210409]_[Learner@reg_xgb_tree]_[Id@84].[Mean0.438318]_[Std0.000786].csv).

#####3. Best Ensemble Model
After you have built `some` 1st level models, run the folliwng command to generate the best ensemble model:
`python run_stacking_ridge.py -l 2 -d 0 -t 10 -c 1 -L reg_ensemble`

This should generate a submission with local CV RMSE around ~0.436. (The best ensemble model we have generated is [here](./Output/Subm/test.pred.[Feat@level2_meta_linear_201605030922]_[Learner@reg_ensemble]_[Id@1].[Mean0.436087]_[Std0.001027].csv).

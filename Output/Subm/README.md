0. sub0: `test.pred.[Feat@basic_nonlinear_201604210409]_[Learner@reg_xgb_tree]_[Id@84].[Mean0.438318]_[Std0.000786].csv`
 - reproduced best single model from Chenglong
 - Public LB: **0.43996**
 - Private LB: **0.43811** (9th place)

1. sub1: `submission_kostia + igor final_ensemble (1 to 3 weights).csv`
 - reproduced best ensembled model from Igor and Kostia
 - Public LB: **0.43819**
 - Private LB: **0.43704** (8th place)

2. sub2: `test.pred.[Feat@level2_meta_linear_201605030922]_[Learner@reg_ensemble]_[Id@1].[Mean0.436087]_[Std0.001027].csv`
 - reproduced best ensembled model from Chenglong
 - Public LB: **0.43582**
 - Private LB: **0.43325** (4th place)

3. sub3: `reproduced_blend_0.438_0.436CV.csv`
 - reproduced best blended model from 0.3 * sub1 + 0.7 * sub2
 - Public LB: **0.43465**
 - Private LB: **0.43248** (3rd place)
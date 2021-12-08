#!/bin/bash

wait_time=0s
echo `date`, "start after ${wait_time}"
sleep ${wait_time}
echo `date`, "start task"

## ------------------- for those freeze ---------------------
# python -m models.freeze_trainer --cfg configs/exps_freeze/exps_cuhk.closs_45.validate.yml
# python -m evaluation.eval_graph exps/exps_freeze/exps_cuhk.closs_45.validate

# python -m models.freeze_trainer --cfg configs/exps_freeze/exps_prw.closs_50.validate.yml
# python -m evaluation.eval_graph exps/exps_freeze/exps_prw.closs_50.validate


## ------------------- for those from scratch ---------------------
# python -m models.graph_trainer --cfg configs/exps_acae/exps_prw.closs_60.validate.yml
# python -m evaluation.eval_graph exps/exps_acae/exps_prw.closs_60.validate

## ------------------- for those from scratch again ---------------------
# python -m models.graph_trainer --cfg configs/exps_acae/exps_cuhk.closs_35.validate.yml
# python -m evaluation.eval_graph exps/exps_acae/exps_cuhk.closs_35.validate

python -m models.graph_trainer --cfg configs/exps_acae/exps_cuhk.closs_35.new_validate.yml
python -m evaluation.eval_graph exps/exps_acae/exps_cuhk.closs_35.new_validate

echo `date`, "task finised"

#!/bin/bash

wait_time=0s
echo `date`, "start after ${wait_time}"
sleep ${wait_time}
echo `date`, "start task"

## ----------------- CUHK ------------------------
# train
# python -m models.freeze_trainer --cfg configs/exps_freeze/exps_cuhk.yml

# validate that model is fixed.
# python -m evaluation.eval_graph exps/exps_freeze/exps_cuhk

# validate
# python -m evaluation.eval_graph exps/exps_freeze/exps_cuhk \
#     --eval-config exps/exps_freeze/exps_cuhk/eval.validate.yml

## ----------------- CUHK ------------------------
# # train
# python -m models.freeze_trainer --cfg configs/exps_freeze/exps_cuhk.closs.test.yml
# python -m evaluation.eval_graph exps/exps_freeze/exps_cuhk.closs

# python -m models.freeze_trainer --cfg configs/exps_freeze/exps_cuhk.triplet.yml
# python -m evaluation.eval_graph exps/exps_freeze/exps_cuhk.triplet

# python -m models.freeze_trainer --cfg configs/exps_freeze/exps_cuhk.oim.yml
# python -m evaluation.eval_graph exps/exps_freeze/exps_cuhk.oim

# python -m models.freeze_trainer --cfg configs/exps_freeze/exps_cuhk.circle.yml
# python -m evaluation.eval_graph exps/exps_freeze/exps_cuhk.circle

# python -m models.freeze_trainer --cfg configs/exps_freeze/exps_cuhk.closs_35.yml
# python -m evaluation.eval_graph exps/exps_freeze/exps_cuhk.closs_35

# python -m models.freeze_trainer --cfg configs/exps_freeze/exps_cuhk.triplet_35.yml
# python -m evaluation.eval_graph exps/exps_freeze/exps_cuhk.triplet_35

# python -m models.freeze_trainer --cfg configs/exps_freeze/exps_cuhk.closs_45.yml
# python -m evaluation.eval_graph exps/exps_freeze/exps_cuhk.closs_45

# python -m models.freeze_trainer --cfg configs/exps_freeze/exps_cuhk.closs_50.yml
# python -m evaluation.eval_graph exps/exps_freeze/exps_cuhk.closs_50

# python -m models.freeze_trainer --cfg configs/exps_freeze/exps_cuhk.closs_40.yml
# python -m evaluation.eval_graph exps/exps_freeze/exps_cuhk.closs_40

# python -m models.graph_trainer --cfg configs/exps_acae/exps_cuhk.closs_35.yml
# python -m evaluation.eval_graph exps/exps_acae/exps_cuhk.closs_35


echo `date`, "task finised"

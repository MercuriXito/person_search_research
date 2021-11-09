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

# # validate the 
python -m evaluation.eval_graph exps/exps_freeze/exps_cuhk \
    --eval-config exps/exps_freeze/exps_cuhk/eval.validate.yml

echo `date`, "task finised"

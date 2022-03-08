#!/bin/bash

wait_time=0s
echo `date`, "start after ${wait_time}"
sleep ${wait_time}
echo `date`, "start task"

python -m models.trainer --cfg configs/exps_det_configs/exps_fcos/exps_cuhk.yml 
python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fcos

echo `date`, "task finised"

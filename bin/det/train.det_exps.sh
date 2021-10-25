#!/bin/bash

wait_time=0s
echo `date`, "start after ${wait_time}"
sleep ${wait_time}
echo `date`, "start task"

## ----------------- CUHK ------------------------

# train detection models only
# python -m models.trainer --cfg configs/exps_det_configs/exps_fpn.yml
# python -m models.trainer --cfg configs/exps_det_configs/exps_fpn.bn.yml


# train on full person search models
python -m models.trainer --cfg configs/exps_det_configs/exps_cuhk.fpn.bn.yml


echo `date`, "task finised"

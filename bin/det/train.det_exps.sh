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
# python -m models.trainer --cfg configs/exps_det_configs/exps_cuhk.fpn.bn.yml
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fpn

# enable layer1
# python -m models.trainer --cfg configs/exps_det_configs/exps_cuhk.fpn.bn.layer1.yml
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fpn.layer1

# k sampling on oim loss
# python -m models.trainer --cfg configs/exps_det_configs/exps_cuhk.fpn.k32.yml
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fpn.k32

# k sampling on oim loss, k = 16
# python -m models.trainer --cfg configs/exps_det_configs/exps_cuhk.fpn.k16.yml
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fpn.k16

# k sampling on oim loss, k = 8
python -m models.trainer --cfg configs/exps_det_configs/exps_cuhk.fpn.k8.yml
python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fpn.k8

echo `date`, "task finised"

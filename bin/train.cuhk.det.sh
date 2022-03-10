#!/bin/bash

wait_time=0s
echo `date`, "start after ${wait_time}"
sleep ${wait_time}
echo `date`, "start task"

# python -m models.trainer --cfg configs/exps_det_configs/exps_fcos/exps_cuhk.yml 
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fcos

# python -m models.trainer --cfg configs/exps_det_configs/exps_fcos/exps_cuhk.loss_cls_5.yml 
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fcos.loss_cls_5
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fcos.loss_cls_5 eval.use_gt True

python -m models.trainer --cfg configs/exps_det_configs/exps_fcos/exps_cuhk.loss_weights.yml
python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fcos.loss_weights
python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fcos.loss_weights eval.use_gt True

echo `date`, "task finised"

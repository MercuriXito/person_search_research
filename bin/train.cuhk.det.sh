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

# python -m models.trainer --cfg configs/exps_det_configs/exps_fcos/exps_cuhk.loss_weights.yml
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fcos.loss_weights
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fcos.loss_weights eval.use_gt True

# python -m models.trainer --cfg configs/exps_det_configs/exps_fcos/exps_cuhk.loss_cls_5.center.yml 
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fcos.loss_cls_5.center
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fcos.loss_cls_5.center eval.use_gt True

# python -m models.trainer --cfg configs/exps_det_configs/exps_retinanet/exps_cuhk.retinanet.loss_weights.yml
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.retinanet.loss_weights
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.retinanet.loss_weights eval.use_gt True

# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fcos.loss_cls_5.center.iter
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fcos.loss_cls_5.center.iter eval.use_gt True

# python -m models.iter_trainer --cfg configs/exps_det_configs/exps_retinanet/exps_cuhk.retinanet.loss_weights.iter.yml
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.retinanet.loss_weights.iter
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.retinanet.loss_weights.iter eval.use_gt True

# python -m models.iter_trainer --cfg configs/exps_det_configs/exps_retinanet/exps_cuhk.retinanet.loss_weights.iter.new_head.yml
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.retinanet.loss_weights.iter.new_head
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.retinanet.loss_weights.iter.new_head eval.use_gt True

# python -m models.iter_trainer --cfg configs/exps_det_configs/exps_retinanet/exps_cuhk.retinanet.loss_weights.iter.new_head.large1024.yml
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.retinanet.loss_weights.iter.new_head.large1024
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.retinanet.loss_weights.iter.new_head.large1024 eval.use_gt True

# python -m models.iter_trainer --cfg configs/exps_det_configs/exps_retinanet/exps_cuhk.retinanet.loss_weights.iter.single.yml
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.retinanet.loss_weights.iter.single
# python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.retinanet.loss_weights.iter.single eval.use_gt True

python -m models.iter_trainer --cfg configs/exps_det_configs/exps_retinanet/exps_cuhk.retinanet.loss_weights.iter.new_head.large2048.yml
python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.retinanet.loss_weights.iter.new_head.large2048
python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.retinanet.loss_weights.iter.new_head.large2048 eval.use_gt True

echo `date`, "task finised"

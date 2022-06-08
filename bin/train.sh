#!/bin/bash

wait_time=0s
echo `date`, "start after ${wait_time}"
sleep ${wait_time}
echo `date`, "start task"

## ========================== baseline experiments ==============================
python -m models.trainer --cfg configs/exps_configs/exps_cuhk.yml
python -m evaluation.eval_defaults exps/exps_cuhk
python -m evaluation.eval_defaults exps/exps_cuhk  \  # or modify config.yml to eval.yml
    eval.eval_context True \
    eval.graph_thred 0.4 \

# baseline on PRW
python -m models.trainer --cfg configs/exps_configs/exps_prw.yml
python -m evaluation.eval_defaults exps/exps_prw
python -m evaluation.eval_defaults exps/exps_prw \  # or modify config.yml to eval.yml
    eval.eval_context True \
    eval.graph_thred 0.4 \


## ========================== ACAE experiments ==============================
python -m models.graph_trainer --cfg configs/exps_acae/exps_cuhk.closs_35.yml
python -m evaluation.eval_graph exps/exps_acae/exps_cuhk.closs_35
python -m evaluation.eval_graph exps/exps_acae/exps_cuhk.closs_35 \
    eval.eval_context True \
    eval.eval_all_sim True \
    eval.graph_thred 0.4 \

# ACAE on PRW
python -m models.graph_trainer --cfg configs/exps_acae/exps_prw.closs_60.yml
python -m evaluation.eval_graph exps/exps_acae/exps_prw.closs_60
python -m evaluation.eval_graph exps/exps_acae/exps_prw.closs_60 \
    eval.eval_context True \
    eval.eval_all_sim True \
    eval.graph_thred 0.4 \


## ====================== train on different det-based models ================
python -m models.trainer --cfg configs/exps_det_configs/exps_fpn/exps_cuhk.fpn.bn.layer1.yml
python -m models.iter_trainer --cfg configs/exps_det_configs/exps_fcos/exps_cuhk.loss_cls_5.center.iter.yml
python -m models.iter_trainer --cfg configs/exps_det_configs/exps_retinanet/exps_cuhk.retinanet.loss_weights.iter.new_head.large2048.yml

echo `date`, "task finised"

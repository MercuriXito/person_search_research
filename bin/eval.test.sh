#!/bin/bash

# evaluation without eval_context
python -m evaluation.eval_defaults exps/exps_cuhk \
    eval.eval_context False

python -m evaluation.eval_defaults exps/exps_prw \
    eval.eval_context False

python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fcos.loss_cls_5.center.iter \
    eval.eval_context False
    
python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fpn.layer1 \
    eval.eval_context False

python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.retinanet.loss_weights.iter.new_head.large2048 \
    eval.eval_context False

# evaluation with eval_context
python -m evaluation.eval_defaults exps/exps_cuhk 
python -m evaluation.eval_defaults exps/exps_prw 
python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fcos.loss_cls_5.center.iter 
python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.fpn.layer1 
python -m evaluation.eval_defaults exps/exps_det/exps_cuhk.retinanet.loss_weights.iter.new_head.large2048 

# evalution without ACAE
python -m evaluation.eval_graph exps/exps_acae/exps_cuhk.closs_35 \
    eval.eval_context False

python -m evaluation.eval_graph exps/exps_acae/exps_prw.closs_60 \
    eval.eval_context False

# evalution with ACAE
python -m evaluation.eval_graph exps/exps_acae/exps_cuhk.closs_35
python -m evaluation.eval_graph exps/exps_acae/exps_prw.closs_60

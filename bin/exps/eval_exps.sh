#!/bin/bash

# =============================================================
# CMM experiments: Ablation
# =============================================================
# python -m experiments.exps_cmm_ablation --pickle exps/exps_cuhk/checkpoint.pth.eval.pkl

# =============================================================
# CMM experiments: sensity of lambda
# =============================================================
# python -m experiments.exps_lambda --pickle exps/exps_cuhk/checkpoint.pth.eval.pkl

# =============================================================
# CMM experiments: gallery size effects.
# =============================================================
# python -m experiments.exps_gallery_size --pickle exps/exps_cuhk/checkpoint.pth.eval.pkl

# =============================================================
# CMM experiments: CMM on other methods.
# =============================================================
# RetinaNet-PS
# python -m evaluation.eval_defaults \
#     exps/exps_det/exps_cuhk.retinanet.loss_weights.iter.new_head.large2048
# python -m evaluation.eval_defaults \
#     exps/exps_det/exps_cuhk.retinanet.loss_weights.iter.new_head.large2048 \
#     eval.use_gt True

# FCOS-PS
# python -m evaluation.eval_defaults \
#     exps/exps_det/exps_cuhk.fcos.loss_cls_5.center.iter
# python -m evaluation.eval_defaults \
#     exps/exps_det/exps_cuhk.fcos.loss_cls_5.center.iter \
#     eval.use_gt True

# FPN-PS orignal use_gt
# python -m evaluation.eval_defaults \
#     --eval-config exps/exps_det/exps_cuhk.fpn/config.yml \
#     exps/exps_det/exps_cuhk.fpn \
#     eval.use_gt True

# FPN-PS
# python -m evaluation.eval_defaults \
#     exps/exps_det/exps_cuhk.fpn \
#     eval.use_gt True

# # baseline
# python -m evaluation.eval_defaults \
#     exps/exps_cuhk \
#     eval.use_gt True

# =============================================================
# ACAE experiments: sensity of lambda
# =============================================================
# python -m experiments.exps_lambda --pickle exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.eval.validate.pkl

# =============================================================
# ACAE experiments: options of different features
# =============================================================
# python -m experiments.exps_acae_ablation --pickle exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.eval.validate.pkl

# =============================================================
# ACAE experiments: gallery size effects.
# =============================================================
# python -m experiments.exps_gallery_size --pickle exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.eval.validate.pkl

# =============================================================
# ACAE experiments: Topk Effects.
# =============================================================
# python -m experiments.exps_fast_graph --pickle exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.eval.validate.pkl
# python -m experiments.exps_fast_graph --pickle exps/exps_acae/exps_prw.closs_60/checkpoint.pth.acae.G0.4.eval.pkl

# =============================================================
# Inference Time Measurement.
# =============================================================
# python -m experiments.exps_speed --pickle exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.eval.validate.pkl
# python -m experiments.exps_speed --pickle exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.eval.validate.pkl --mode feat
# python -m experiments.exps_speed --pickle exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.eval.validate.pkl --mode baseline
# python -m experiments.exps_speed --pickle exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.eval.validate.pkl --mode cmm
# python -m experiments.exps_speed --pickle exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.eval.validate.pkl --mode acae

# Inference time on other-ps model
# python -m experiments.exps_speed --pickle exps/exps_det_acae/exps_cuhk.retinanet.freeze/checkpoint.pth.acae.G0.4.eval.pkl --mode feat
# python -m experiments.exps_speed --pickle exps/exps_det_acae/exps_cuhk.retinanet.freeze/checkpoint.pth.acae.G0.4.eval.pkl --mode baseline
# python -m experiments.exps_speed --pickle exps/exps_det_acae/exps_cuhk.retinanet.freeze/checkpoint.pth.acae.G0.4.eval.pkl --mode cmm
# python -m experiments.exps_speed --pickle exps/exps_det_acae/exps_cuhk.retinanet.freeze/checkpoint.pth.acae.G0.4.eval.pkl --mode acae

# python -m experiments.exps_speed --pickle exps/exps_det_acae/exps_cuhk.fcos.freeze/checkpoint.pth.acae.G0.4.eval.pkl --mode feat
# python -m experiments.exps_speed --pickle exps/exps_det_acae/exps_cuhk.fcos.freeze/checkpoint.pth.acae.G0.4.eval.pkl --mode baseline
# python -m experiments.exps_speed --pickle exps/exps_det_acae/exps_cuhk.fcos.freeze/checkpoint.pth.acae.G0.4.eval.pkl --mode cmm
# python -m experiments.exps_speed --pickle exps/exps_det_acae/exps_cuhk.fcos.freeze/checkpoint.pth.acae.G0.4.eval.pkl --mode acae

# python -m experiments.exps_speed --pickle exps/exps_det_acae/exps_cuhk.fpn.freeze/checkpoint.pth.acae.G0.4.eval.pkl --mode feat
# python -m experiments.exps_speed --pickle exps/exps_det_acae/exps_cuhk.fpn.freeze/checkpoint.pth.acae.G0.4.eval.pkl --mode baseline
# python -m experiments.exps_speed --pickle exps/exps_det_acae/exps_cuhk.fpn.freeze/checkpoint.pth.acae.G0.4.eval.pkl --mode cmm
# python -m experiments.exps_speed --pickle exps/exps_det_acae/exps_cuhk.fpn.freeze/checkpoint.pth.acae.G0.4.eval.pkl --mode acae

# =============================================================
# Addition Inference Time Measurement.
# =============================================================

# ACAE
# python -m evaluation.eval_graph \
#     exps/exps_acae/exps_cuhk.closs_35 \
#     eval.use_data exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.eval.validate.pkl

# # CMM
# python -m evaluation.eval_defaults \
#     exps/exps_cuhk \
#     eval.use_data exps/exps_cuhk/checkpoint.pth.ctx.G0.4.eval.pkl

# # ACAE on PRW
# python -m evaluation.eval_graph \
#     exps/exps_acae/exps_prw.closs_60 \
#     eval.use_data exps/exps_acae/exps_prw.closs_60/checkpoint.pth.acae.G0.4.eval.pkl

# # CMM on PRW
# python -m evaluation.eval_defaults \
#     exps/exps_prw.oim.graph.lossw_101 \
#     eval.use_data exps/exps_prw.oim.graph.lossw_101/checkpoint.pth.ctx.G0.4.eval.pkl

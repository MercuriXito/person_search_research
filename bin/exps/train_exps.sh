#!/bin/bash

wait_time=0h
echo `date`, "start after ${wait_time}"
sleep ${wait_time}
echo `date`, "start task"

####################################################
# Experiments: training without using lut          #
####################################################

# python -m models.pair_graph_trainer --cfg configs/exps_acae/exps_cuhk.closs_35.wo_lut.yml
# python -m evaluation.eval_graph \
#     --eval-config exps/exps_acae/exps_cuhk.closs_35.wo_lut/config.yml \
#     exps/exps_acae/exps_cuhk.closs_35.wo_lut
# python -m evaluation.eval_graph exps/exps_acae/exps_cuhk.closs_35.wo_lut

######################################################################
# Experiments: ACAE on different ps model: freeze training.          #
######################################################################

#####  Experiments on retinanet

# python -m models.freeze_trainer --cfg configs/exps_det_configs/exps_acae/exps_cuhk.retinanet.freeze.yml

# python -m evaluation.eval_graph \
#     --eval-config exps/exps_det_acae/exps_cuhk.retinanet.freeze/config.yml \
#     exps/exps_det_acae/exps_cuhk.retinanet.freeze

# python -m evaluation.eval_graph exps/exps_det_acae/exps_cuhk.retinanet.freeze

#####  Experiments on fcos

# python -m models.freeze_trainer --cfg configs/exps_det_configs/exps_acae/exps_cuhk.fcos.freeze.yml

# python -m evaluation.eval_graph \
#     --eval-config exps/exps_det_acae/exps_cuhk.fcos.freeze/config.yml \
#     exps/exps_det_acae/exps_cuhk.fcos.freeze

# python -m evaluation.eval_graph exps/exps_det_acae/exps_cuhk.fcos.freeze

#####  Experiments on fpn

# python -m models.freeze_trainer --cfg configs/exps_det_configs/exps_acae/exps_cuhk.fpn.freeze.yml
python -m evaluation.eval_graph exps/exps_det_acae/exps_cuhk.fpn.freeze

####################################################
# Experiments: ACAE on different ps model          #
####################################################

#####  Experiments on retinanet

# python -m models.graph_trainer --cfg configs/exps_det_configs/exps_acae/exps_cuhk.retinanet.yml

# python -m evaluation.eval_graph \
#     --eval-config exps/exps_det_acae/exps_cuhk.retinanet/config.yml \
#     exps/exps_det_acae/exps_cuhk.retinanet

# python -m evaluation.eval_graph exps/exps_det_acae/exps_cuhk.retinanet

### RetinaNet-ACAE hyper-parameters
# python -m models.graph_trainer --cfg configs/exps_det_configs/exps_acae/exps_cuhk.retinanet.lw1.yml
# python -m evaluation.eval_graph exps/exps_det_acae/exps_cuhk.retinanet.lw1

# python -m models.graph_trainer --cfg configs/exps_det_configs/exps_acae/exps_cuhk.retinanet.iter.yml
# python -m evaluation.eval_graph exps/exps_det_acae/exps_cuhk.retinanet.iter

### RetinaNet-ACAE with iter_trainer
# python -m models.graph_trainer --cfg configs/exps_det_configs/exps_acae/exps_cuhk.retinanet.iter.yml
# python -m evaluation.eval_graph exps/exps_det_acae/exps_cuhk.retinanet.iter

#####  Experiments on fcos

# python -m models.graph_trainer --cfg configs/exps_det_configs/exps_acae/exps_cuhk.fcos.iter.yml
# python -m evaluation.eval_graph exps/exps_det_acae/exps_cuhk.fcos.iter

# python -m evaluation.eval_graph \
#     exps/exps_det_acae/exps_cuhk.fcos.iter \
#     eval.use_gt True

# python -m evaluation.eval_graph \
#     exps/exps_det_acae/exps_cuhk.retinanet.iter \
#     eval.use_gt True

##### Experiment on fpn

# python -m models.graph_trainer --cfg configs/exps_det_configs/exps_acae/exps_cuhk.fpn.yml
# python -m evaluation.eval_graph exps/exps_det_acae/exps_cuhk.fpn
# python -m evaluation.eval_graph \
#     exps/exps_det_acae/exps_cuhk.fpn \
#     eval.use_gt True

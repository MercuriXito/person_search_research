#!/bin/bash

wait_time=0s
echo `date`, "start after ${wait_time}"
sleep ${wait_time}
echo `date`, "start task"

# run command
python -m evaluation.eval_defaults \
    exps/exps_det/exps_cuhk.retinanet.fpn_reid_head.sample \
    eval.checkpoint "checkpoint0020.pth"

# python -m evaluation.eval_defaults \
#     exps/exps_det/exps_cuhk.retinanet.fpn_reid_head.sample \
#     eval.checkpoint "checkpoint0005.pth"

python -m evaluation.eval_defaults \
    exps/exps_det/exps_cuhk.retinanet.fpn_reid_head.sample \
    eval.checkpoint "checkpoint0010.pth"

python -m evaluation.eval_defaults \
    exps/exps_det/exps_cuhk.retinanet.fpn_reid_head.sample \
    eval.checkpoint "checkpoint0015.pth"

echo `date`, "task finised"

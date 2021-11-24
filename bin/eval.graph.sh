#!/bin/bash

wait_time=0s
echo `date`, "start after ${wait_time}"
sleep ${wait_time}
echo `date`, "start task"

# run command
# python -m evaluation.eval_graph exps/exps_cuhk.graph
# python -m evaluation.eval_graph exps/exps_cuhk.graph.lossw_21
# python -m evaluation.eval_graph exps/exps_prw.oim.graph

python -m evaluation.eval_graph exps/exps_acae/exps_prw.closs_35

echo `date`, "task finised"

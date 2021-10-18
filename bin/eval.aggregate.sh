#!/bin/bash 

# python -m evaluation.eval_defaults exps/exps_cuhk.graph.lossw_101
python -m evaluation.eval_aggregate exps/exps_cuhk.graph.lossw_101 \
    --eval-cfg exps/exps_cuhk.graph.lossw_101/cmm_eval.yml


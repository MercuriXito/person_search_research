#!/bin/bash

wait_time=0s
echo `date`, "start after ${wait_time}"
sleep ${wait_time}
echo `date`, "start task"

## ----------------- CUHK ------------------------

# train
# python -m models.graph_trainer --cfg configs/exps_configs/exps_cuhk.graph.yml

# experiment: lossw_21
# python -m models.graph_trainer --cfg configs/exps_configs/exps_cuhk.graph.lossw_21.yml

# experiment: lossw_101
# python -m models.graph_trainer --cfg configs/exps_configs/exps_cuhk.graph.lossw_101.yml

# experiment: use ema_lut
# python -m models.graph_trainer --cfg configs/exps_configs/exps_cuhk.graph.lossw_101.ema.yml
python -m evaluation.eval_graph exps/exps_cuhk.graph.lossw_101.ema

## ----------------- PRW ------------------------
# PRW graph baseline
# python -m models.graph_trainer --cfg configs/exps_configs/exps_prw.graph.yml

# PRW graph lossw101
# python -m models.graph_trainer --cfg configs/exps_configs/exps_prw.graph.lossw_101.yml
# python -m evaluation.eval_graph exps/exps_prw.oim.graph.lossw_101


echo `date`, "task finised"

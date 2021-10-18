#!/bin/bash


# main_diff_map

# # comapre the ap of each query between two models.
# python -m evaluation.badcase \
#     --pkl1 exps/exps_cuhk/checkpoint.pth.ctx.G0.4.eval.pkl \
#     --pkl2 exps/exps_cuhk.graph.lossw_101/checkpoint.pth.ctx.G0.4.eval.pkl

# # comapre the ap of each query between two models.
# python -m evaluation.badcase \
#     --pkl1 exps/exps_cuhk.graph.lossw_101/checkpoint.pth.ctx.G0.4.eval.pkl \
#     --pkl2 exps/exps_cuhk/checkpoint.pth.ctx.G0.4.eval.pkl


# main_aps

# python -m evaluation.badcase \
#     --pkl exps/exps_cuhk/checkpoint.pth.ctx.G0.4.eval.pkl

# python -m evaluation.badcase \
#     --pkl exps/exps_cuhk/checkpoint0019.pth.eval.pkl

python -m evaluation.badcase \
    --pkl exps/exps_cuhk.graph.lossw_101/checkpoint.pth.eval.pkl

# python -m evaluation.badcase \
#     --pkl exps/exps_cuhk.graph.lossw_101/checkpoint.pth.ctx.G0.4.eval.pkl

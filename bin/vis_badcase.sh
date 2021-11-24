#!/bin/bash

# ---------------------- main_diff_map

# # comapre the ap of each query between two models.
# python -m evaluation.badcase \
#     --pkl1 exps/exps_cuhk/checkpoint.pth.ctx.G0.4.eval.pkl \
#     --pkl2 exps/exps_cuhk.graph.lossw_101/checkpoint.pth.ctx.G0.4.eval.pkl

# # comapre the ap of each query between two models.
# python -m evaluation.badcase \
#     --pkl1 exps/exps_cuhk.graph.lossw_101/checkpoint.pth.ctx.G0.4.eval.pkl \
#     --pkl2 exps/exps_cuhk/checkpoint.pth.ctx.G0.4.eval.pkl

python -m evaluation.badcase \
    --pkl1 exps/exps_cuhk/checkpoint.pth.ctx.G0.4.eval.pkl \
    --pkl2 exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.eval.pkl \

python -m evaluation.badcase \
    --pkl1 exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.eval.pkl \
    --pkl2 exps/exps_cuhk/checkpoint.pth.ctx.G0.4.eval.pkl \

# -------------------------- main_aps

# python -m evaluation.badcase \
#     --pkl exps/exps_cuhk/checkpoint.pth.ctx.G0.4.eval.pkl

# python -m evaluation.badcase \
#     --pkl exps/exps_cuhk.graph.lossw_101/checkpoint.pth.eval.pkl

# python -m evaluation.badcase \
#     --pkl exps/exps_cuhk.graph.lossw_101/checkpoint.pth.ctx.G0.4.eval.pkl

# python -m evaluation.badcase \
#     --pkl exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.eval.pkl

# ---------------------------- visualize_ap_worse_samples
# python -m evaluation.badcase \
#      --pkl1 exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.eval.pkl \
#      --pkl2 exps/exps_cuhk/checkpoint.pth.ctx.G0.4.eval.pkl

# python -m evaluation.badcase \
#      --pkl1 exps/exps_cuhk/checkpoint.pth.ctx.G0.4.eval.pkl \
#      --pkl2 exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.eval.pkl \


# ------------------------------- visualize attention map
# python -m evaluation.vis_attn \
#     --pkl exps/exps_acae/exps_cuhk.closs_35/checkpoint.pth.acae.G0.4.eval.pkl \


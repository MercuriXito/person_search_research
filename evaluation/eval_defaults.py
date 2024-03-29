import torch
import os

from utils.misc import pickle
from models import build_graph_models, build_models
from configs.faster_rcnn_default_configs import get_default_cfg
from evaluation.eval import FasterRCNNExtractor, evaluate, \
    GTFeatureExtractor
from evaluation.evaluator import PersonSearchEvaluator


def build_and_load_from_dir(exp_dir, dst_eval_file="", other_options=None):
    """ load fasterrcnn based model configs from saved folder:
        - resume option from config.yml
        - build model, resume from checkpoint
        - resume evaluation options
    """
    if other_options is None:
        other_options = []

    # priority to choose evaluation file
    # designated eval file -> eval.yml -> config.yml
    if len(dst_eval_file) > 0:
        eval_file = dst_eval_file
        assert os.path.exists(eval_file), f"{eval_file}"
    else:
        eval_file = os.path.join(exp_dir, "eval.yml")

    if os.path.exists(eval_file):
        config_file = eval_file
    else:
        config_file = os.path.join(exp_dir, "config.yml")
    print(f"Using {config_file} for evaluation.")

    t_args = get_default_cfg()
    t_args.merge_from_file(config_file)
    t_args.merge_from_list(other_options)
    t_args.freeze()
    eval_args = t_args.eval
    print(eval_args)

    # load model
    if t_args.model.graph_head.use_graph:
        model = build_graph_models(t_args)
    else:
        model = build_models(t_args)

    # HACK: checkpoint
    checkpoint_path = os.path.join(exp_dir, eval_args.checkpoint)
    params = torch.load(checkpoint_path, map_location="cpu")
    model_params = params["model"]
    missed, unexpected = model.load_state_dict(model_params, strict=False)
    if len(unexpected) > 0:
        print(f"Unexpected keys: {unexpected}")
    if len(missed) > 0:
        print(f"Missed keys: {missed}")

    return model, t_args


def main():
    """ Default evaluation with baseline network.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    parser.add_argument("--eval-config", default="", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    model, t_args = build_and_load_from_dir(args.exp_dir, args.eval_config, args.opts)
    eval_args = t_args.eval
    device = torch.device(eval_args.device)
    if eval_args.use_gt:
        extractor = GTFeatureExtractor(model, device)
    else:
        extractor = FasterRCNNExtractor(model, device)
    ps_evaluator = PersonSearchEvaluator(eval_args.dataset_file)
    res_pkl, table_string = evaluate(extractor, eval_args, ps_evaluator=ps_evaluator)

    checkpoint_path = os.path.join(args.exp_dir, eval_args.checkpoint)
    # serealization
    prefix = "eval"
    if eval_args.eval_context:
        prefix = f"cmm.G{eval_args.graph_thred}.{prefix}"
    if eval_args.use_gt:
        prefix = f"{prefix}.gt"
    save_path = f"{checkpoint_path}.{prefix}.pkl"
    table_string_path = f"{checkpoint_path}.{prefix}.txt"
    pickle(res_pkl, save_path)
    with open(table_string_path, "w") as f:
        f.write(table_string)


if __name__ == '__main__':
    main()

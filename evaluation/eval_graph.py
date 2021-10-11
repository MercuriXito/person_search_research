import torch
import os

from evaluation.evaluator import GraphPSEvaluator
from evaluation.eval import FasterRCNNExtractor
from models.graph_net import build_graph_net


def main():
    """ Evaluation with GraphNet which includes ACAE branch.
    """
    import argparse
    from configs.graph_net_default_configs import get_default_cfg
    from evaluation.eval import evaluate
    from utils.misc import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    args = parser.parse_args()

    eval_file = os.path.join(args.exp_dir, "eval.yml")
    if os.path.exists(eval_file):
        config_file = eval_file
    else:
        config_file = os.path.join(args.exp_dir, "config.yml")

    t_args = get_default_cfg()
    t_args.merge_from_file(config_file)
    t_args.freeze()
    eval_args = t_args.eval

    # load model
    model = build_graph_net(t_args)

    # HACK: checkpoint
    checkpoint_path = os.path.join(args.exp_dir, eval_args.checkpoint)
    params = torch.load(checkpoint_path, map_location="cpu")
    model_params = params["model"]
    missed, unexpected = model.load_state_dict(model_params, strict=False)
    if len(unexpected) > 0:
        print(f"Unexpected keys: {unexpected}")
    if len(missed) > 0:
        print(f"Missed keys: {missed}")

    device = torch.device(eval_args.device)
    extractor = FasterRCNNExtractor(model, device)
    ps_evaluator = GraphPSEvaluator(model.graph_head, device, eval_args.dataset_file)
    res_pkl, table_string = evaluate(extractor, eval_args, ps_evaluator)

    # serealization
    prefix = "eval"
    if eval_args.eval_context:
        prefix = f"ctx.G{eval_args.graph_thred}.{prefix}"
    save_path = f"{checkpoint_path}.{prefix}.pkl"
    table_string_path = f"{checkpoint_path}.{prefix}.txt"
    pickle(res_pkl, save_path)
    with open(table_string_path, "w") as f:
        f.write(table_string)


if __name__ == '__main__':
    main()

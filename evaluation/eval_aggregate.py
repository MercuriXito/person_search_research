import torch
import os

from utils.misc import pickle
from evaluation.evaluator import AggregatedPSEvaluator
from evaluation.eval import FasterRCNNExtractor, evaluate
from evaluation.eval_defaults import build_and_load_from_dir
from evaluation.eval_args import get_eval_args


def main():
    """ Evaluation with GraphNet which includes ACAE branch.
    """
    args = get_eval_args()

    model, t_args = build_and_load_from_dir(args.exp_dir, args.eval_cfg)
    eval_args = t_args.eval
    checkpoint_path = os.path.join(args.exp_dir, eval_args.checkpoint)

    device = torch.device(eval_args.device)
    extractor = FasterRCNNExtractor(model, device)
    ps_evaluator = AggregatedPSEvaluator(model.graph_head, device, eval_args.dataset_file)
    res_pkl, table_string = evaluate(extractor, eval_args, ps_evaluator=ps_evaluator)

    # serealization
    prefix = "eval"
    if eval_args.eval_context:
        prefix = f"agg.G{eval_args.graph_thred}.{prefix}"
    save_path = f"{checkpoint_path}.{prefix}.pkl"
    table_string_path = f"{checkpoint_path}.{prefix}.txt"
    pickle(res_pkl, save_path)
    with open(table_string_path, "w") as f:
        f.write(table_string)


if __name__ == '__main__':
    main()

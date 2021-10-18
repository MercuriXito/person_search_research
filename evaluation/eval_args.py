import os
import argparse


def get_eval_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    parser.add_argument("--eval-cfg", default="", type=str)
    return parser


def get_eval_args():
    """ Load params and process the args.
    """
    args = get_eval_parser().parse_args()

    # priority to choose evaluation file
    # designated eval file -> eval.yml -> config.yml
    exp_dir = args.exp_dir
    if len(args.eval_cfg) == 0:
        if "eval.yml" in os.listdir(args.exp_dir):
            eval_file = os.path.join(exp_dir, "eval.yml")
        else:
            eval_file = os.path.join(exp_dir, "config.yml")
    else:
        eval_file = args.eval_cfg

    assert os.path.exists(eval_file)
    args.eval_cfg = eval_file
    return args

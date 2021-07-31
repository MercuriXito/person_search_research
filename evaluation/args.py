import argparse


def get_eval_argparser():
    parser = argparse.ArgumentParser("Evaluation Parser.")

    parser.add_argument("--use-data", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument(
        "--dataset-file", default="cuhksysu",
        choices=["cuhksysu", "prw"], type=str)
    parser.add_argument("--dataset-path", default="")

    parser.add_argument("--gallery-size", default=100, type=int)
    parser.add_argument("--det-thresh", default=0.5, type=float)
    parser.add_argument("--nms-thresh", default=0.4, type=float)
    return parser

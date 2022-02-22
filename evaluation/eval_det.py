import os
import PIL.Image as Image
from prettytable import PrettyTable
from tqdm import tqdm
import torch

from models import build_models
from configs.faster_rcnn_default_configs import get_default_cfg
from utils.misc import ship_to_cuda
from datasets import load_eval_datasets
from evaluation.eval import Person_Search_Features_Extractor


class BoxesExtractor(Person_Search_Features_Extractor):
    """ BoxesExtractor, used especially for evaluating the detection
    performance.
    """
    def __init__(self, model, device) -> None:
        super().__init__(model, device=device)

    def get_gallery_boxes(self, galleries, *args, **kwargs):
        gallery_rois = []

        for item in tqdm(galleries):
            img_path = item["path"]
            image = Image.open(img_path)
            image = [self.transform(image)]
            image = ship_to_cuda(image, device=self.device)

            # outputs = self.model(image)
            outputs = self.model.extract_features_without_boxes(image)
            outputs = outputs[0]
            boxes, scores = outputs["boxes"], outputs["scores"]

            # boxes
            scores = scores.view(-1, 1)
            rois = torch.cat([boxes, scores], dim=1)
            rois = rois.detach().cpu().numpy()
            gallery_rois.append(rois)
        return gallery_rois


def evaluate_detection(extractor, args, imdb=None):
    if imdb is None:
        imdb = load_eval_datasets(args)
    gallery_boxes = extractor.get_gallery_boxes(imdb.roidb, nms_thresh=args.nms_thresh)
    det_ap, det_recall = imdb.detection_performance_calc(
        gallery_boxes, det_thresh=args.det_thresh,
        iou_thresh=args.nms_thresh,
        labeled_only=False)
    label_det_ap, label_det_recall = imdb.detection_performance_calc(
        gallery_boxes, det_thresh=args.det_thresh,
        iou_thresh=args.nms_thresh,
        labeled_only=True)

    table = PrettyTable(field_names=[
        "item", "det_ap", "det_recall", "labeled_ap", "labeled_recall"])
    eval_res = [det_ap, det_recall, label_det_ap, label_det_recall]
    format_eval_res = ["{:.8f}".format(res_item) for res_item in eval_res]
    format_eval_res = ["item"] + format_eval_res
    table.add_row(format_eval_res)
    print(table)


def main():
    # only used for evaluation fpn-detection model.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    args = parser.parse_args()

    t_args = get_default_cfg()
    model_args = dict(
        pretrained=False,
        num_classes=2,
    )
    # HACK configuration.
    model = build_models(t_args)
    model.eval()
    model.load_state_dict(
        torch.load(
            os.path.join(args.exp_dir, "checkpoint.pth"),
            map_location="cpu")["model"]
    )

    eval_args = t_args.eval
    device = torch.device(eval_args.device)
    extractor = BoxesExtractor(model, device)
    evaluate_detection(extractor, eval_args)


if __name__ == '__main__':
    main()

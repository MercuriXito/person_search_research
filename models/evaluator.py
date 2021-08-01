import torch
import os
import PIL.Image as Image
from tqdm import tqdm

from evaluation.eval import Person_Search_Features_Extractor
from models.baseline import build_faster_rcnn_based_models
from utils import ship_to_cuda


class FasterRCNNExtractor(Person_Search_Features_Extractor):
    def __init__(self, model, device) -> None:
        super().__init__(model, device=device)

    def get_gallery_features(self, galleries, *args, **kwargs):
        gallery_features = []
        gallery_rois = []

        for item in tqdm(galleries):
            img_path = item["path"]
            image = Image.open(img_path)
            image = [self.transform(image)]
            image = ship_to_cuda(image, device=self.device)

            outputs = self.model.extract_features_without_boxes(image)
            outputs = outputs[0]

            boxes, features, scores = \
                outputs["boxes"], outputs["embeddings"], outputs["scores"]

            scores = scores.view(-1, 1)
            rois = torch.cat([boxes, scores], dim=1)
            rois = rois.detach().cpu().numpy()
            features = features.detach().cpu().numpy()

            gallery_features.append(features)
            gallery_rois.append(rois)

        return gallery_features, gallery_rois

    def get_query_features(self, probes, use_query_ctx_boxes=False, *args, **kwargs):
        query_features = []
        query_rois = []

        for item in tqdm(probes):
            img_path = item["path"]
            image = Image.open(img_path)
            images = [self.transform(image)]

            boxes = item["boxes"]
            scores = torch.as_tensor([1])
            targets = [dict(boxes=boxes, scores=scores)]
            images, targets = ship_to_cuda(images, targets, self.device)

            if use_query_ctx_boxes:
                pass
            else:
                new_targets = targets

            boxes = [t["boxes"] for t in new_targets]
            scores = [t["scores"].view(-1, 1) for t in new_targets]

            # support batch_size=1 only
            boxes = boxes[0]
            scores = scores[0]

            outputs = self.model.extract_features_with_boxes(images, new_targets)
            features = outputs

            rois = torch.cat([boxes, scores], dim=1)
            rois = rois.detach().cpu().numpy()
            features = features.detach().cpu().numpy()

            query_features.append(features)
            query_rois.append(rois)

        return query_features, query_rois


def evaluate():
    import argparse
    from configs.faster_rcnn_default_configs import get_default_cfg
    from evaluation.eval import evaluate
    from utils import pkl_dump

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
    model = build_faster_rcnn_based_models(t_args)

    # HACK: checkpoint
    checkpoint_path = os.path.join(args.exp_dir, "checkpoint.pth")
    params = torch.load(checkpoint_path, map_location="cpu")
    model_params = params["model"]
    missed, unexpected = model.load_state_dict(model_params, strict=False)
    if len(unexpected) > 0:
        print(f"Unexpected keys: {unexpected}")
    if len(missed) > 0:
        print(f"Missed keys: {missed}")

    device = torch.device(t_args.eval.device)

    extractor = FasterRCNNExtractor(model, device)
    res_pkl, table_string = evaluate(extractor, eval_args)

    save_path = checkpoint_path + "eval.pkl"
    table_string_path = checkpoint_path + "eval.txt"
    pkl_dump(res_pkl, save_path)
    with open(table_string_path, "w") as f:
        f.write(table_string_path)


if __name__ == '__main__':
    evaluate()

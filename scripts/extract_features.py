import argparse
import os
import numpy as np
import torch
import h5py

from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputs,
    fast_rcnn_inference_single_image,
)
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from tqdm import tqdm
import cv2
from detectron2.modeling.postprocessing import detector_postprocess


def prepare_model(
    config_path: str = "configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml",
    model_url: str = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl",
    device: str = "cpu",
):
    print(f"Using device: {device}")
    print(f"Using config: {config_path}")
    print(f"Using model: {model_url}")
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = device
    # VG Weight
    cfg.MODEL.WEIGHTS = model_url
    predictor = DefaultPredictor(cfg)
    buffer = torch.Tensor(
        [
            [-37.0, -15.0, 54.0, 32.0],
            [-83.0, -39.0, 100.0, 56.0],
            [-175.0, -87.0, 192.0, 104.0],
            [-359.0, -183.0, 376.0, 200.0],
            [-23.0, -23.0, 40.0, 40.0],
            [-55.0, -55.0, 72.0, 72.0],
            [-119.0, -119.0, 136.0, 136.0],
            [-247.0, -247.0, 264.0, 264.0],
            [-13.0, -35.0, 30.0, 52.0],
            [-35.0, -79.0, 52.0, 96.0],
            [-79.0, -167.0, 96.0, 184.0],
            [-167.0, -343.0, 184.0, 360.0],
        ]
    )
    predictor.model.proposal_generator.anchor_generator.cell_anchors._buffers[
        "0"
    ] = buffer

    return predictor


def per_image_feature_extraction(
    raw_image: np.ndarray,
    predictor: DefaultPredictor,
    num_objects: int = 100,
    score_thresh: float = 0.1,
    nms_thresh: float = 0.5,
):
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]

        # Preprocessing
        image = predictor.aug.get_transform(raw_image).apply_image(raw_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

        # Predict classes and boxes for each proposal.
        (
            pred_class_logits,
            pred_attr_logits,
            pred_proposal_deltas,
        ) = predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box_predictor.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.box_predictor.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]

        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)

        # Note: BUTD uses raw RoI predictions,
        #       we use the predicted boxes instead.
        # boxes = proposal_boxes[0].tensor

        # NMS
        instances, ids = fast_rcnn_inference_single_image(
            boxes,
            probs,
            image.shape[1:],
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            topk_per_image=num_objects,
        )

        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()
        max_attr_prob = max_attr_prob[ids].detach()
        max_attr_label = max_attr_label[ids].detach()
        instances.attr_scores = max_attr_prob
        instances.attr_classes = max_attr_label

        return instances, roi_features


def prepare_data(data_path: str = "demo/data/genome/1600-400-20"):
    # Load VG Classes
    vg_classes = []
    with open(os.path.join(data_path, "objects_vocab.txt")) as f:
        for object in f.readlines():
            vg_classes.append(object.split(",")[0].lower().strip())

    vg_attrs = []
    with open(os.path.join(data_path, "attributes_vocab.txt")) as f:
        for object in f.readlines():
            vg_attrs.append(object.split(",")[0].lower().strip())

    MetadataCatalog.get("vg").thing_classes = vg_classes
    MetadataCatalog.get("vg").attr_classes = vg_attrs


def extract_features(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Get model
    predictor = prepare_model(device=device.type)
    # Prepare data
    prepare_data()
    # Generate features
    with h5py.File(args.save_features_path, "w") as features:
        for image_name in tqdm(os.listdir(args.flickr30k_images_path)):
            image_path = os.path.join(args.flickr30k_images_path, image_name)
            raw_image = cv2.imread(image_path)
            instances, roi_features = per_image_feature_extraction(
                raw_image, predictor, args.num_objects, args.score_thresh
            )
            image_id = image_name.split(".")[0]
            grp = features.create_group(image_id)
            for attr in [
                "scores",
                "pred_boxes",
                "pred_classes",
                "attr_scores",
                "attr_classes",
            ]:
                value = getattr(instances, attr)
                if not isinstance(value, torch.Tensor):
                    value = value.tensor
                value = value.cpu().numpy()
                # if attr == "pred_classes":
                #     value = np.array(
                #         [
                #             MetadataCatalog.get("vg").thing_classes[v]
                #             for v in value.tolist()
                #         ]
                #     )
                # elif attr == "attr_classes":
                #     value = np.array(
                #         [MetadataCatalog.get("vg").attr_classes[v] for v in value]
                #     )
                # else:
                #     value = value.numpy()
                grp.create_dataset(name=attr, data=value)
            grp.create_dataset(name="roi_features", data=roi_features.cpu().numpy())


def main():
    parser = argparse.ArgumentParser(
        description="Extracts features with a bottom-up attention model"
    )
    parser.add_argument(
        "--flickr30k_images_path",
        type=str,
        required=True,
        help="Path to the Flickr30k images dataset.",
    )
    parser.add_argument(
        "--save_features_path",
        type=str,
        required=True,
        help="Where to save the features.",
    )
    parser.add_argument(
        "--num_objects",
        type=int,
        default=100,
        help="The max number of objects to extract.",
    )
    parser.add_argument(
        "--score_thresh",
        type=int,
        default=0.1,
        help="The per-object confidence threshold.",
    )
    args = parser.parse_args()
    extract_features(args)


if __name__ == "__main__":
    main()

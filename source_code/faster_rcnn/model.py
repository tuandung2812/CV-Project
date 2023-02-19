import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_object_detection_model(num_classes = 2, from_pretrained =  True):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=from_pretrained)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

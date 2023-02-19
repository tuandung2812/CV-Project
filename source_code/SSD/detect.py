from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
import os
import torch
import yaml
from tqdm.notebook import tqdm


def detect_ssd(model, original_image, min_score = 0.2, max_overlap = 0.3, top_k = 200, visualize = False):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    


    original_image = Image.fromarray(original_image)
    original_image = original_image.convert('RGB')


    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')
    # print(det_boxes)
  

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    det_boxes = det_boxes.detach().numpy()
    
    det_scores = det_scores[0].to('cpu').detach().numpy()
    # print(det_scores)
    if visualize :
    # Annotate
      annotated_image = original_image
      draw = ImageDraw.Draw(annotated_image)

      for box in det_boxes:
        x_min, y_min, x_max,y_max  = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        draw.rectangle(((x_min, y_min), (x_max, y_max)), outline = (0,255,0), width= 3)
      return annotated_image
    
    return det_boxes, det_scores
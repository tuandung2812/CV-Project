import torch
from data.wider_face_faster_rcnn import WiderFaceFasterRCNN, detection_collate
from torchvision import transforms
import cv2
import yaml
import os
from tqdm.notebook import tqdm
from faster_rcnn.model import get_object_detection_model
import torch
from torchvision import transforms
import numpy as np
def _resize(image, insize):
    # cv2.imshow("Resized",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image = cv2.resize(image, insize)
    transform = transforms.Compose([transforms.ToTensor()
    ])
    image = transform(image)
    return image




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def faster_rcnn_detect(model, img_path, device, threshold = 0.2, visualize = False):
    img = cv2.imread(img_path)
    original_height, original_width = img.shape[0], img.shape[1]
    # print(self.imgs_path[index])
    # convert BGR to RGB color format
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    image_copy = image
    image = _resize(image, (300,300))
    image = image/ 255.0
    image = torch.unsqueeze(image, 0)
    model.eval()
    with torch.no_grad():
        outputs = model(image.to(device))
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs][0]
    boxes = []
    probabilities = []
    for i in range(len(outputs['scores'])):
      if outputs['scores'][i] >= threshold:
        boxes.append(outputs['boxes'][i].tolist())
        probabilities.append( outputs['scores'][i].tolist())
    
    final_boxes = []
    for box in boxes:
      x_min, y_min, x_max,y_max  = box[0], box[1], box[2], box[3]
      x_min = int( (x_min/300) * original_width )
      y_min = int( (y_min/300) * original_height )
      x_max = int( (x_max/300) * original_width )
      y_max = int( (y_max/300) * original_height )
      final_boxes.append([x_min,y_min,x_max,y_max])
    


    
    if visualize:
      # print(x.shape, x)
      annotated_im = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
      # annotated_im = cv2.resize(image_copy, (300,300))
      for box in final_boxes:
        x_min, y_min, x_max,y_max  = box[0], box[1], box[2], box[3]
        cv2.rectangle(annotated_im, (x_min,y_min), (x_max,y_max), (0,255,0))
      return annotated_im

  
    return  final_boxes, probabilities

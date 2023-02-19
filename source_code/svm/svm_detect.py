import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import cv2
# from google.colab.patches import cv2_imshow
import joblib
from skimage.transform import pyramid_gaussian
import joblib
from PIL import Image
import os
import random
from tqdm.notebook import tqdm
from PIL import Image

def calculate_hog(img):
  cell_size = (6, 6)  # h x w in pixels
  block_size = (2, 2)  # h x w in cells
  nbins = 9  # number of orientation bins

  img = img.astype(np.uint8)
  # print(img)
  hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                    img.shape[0] // cell_size[0] * cell_size[0]),
                          _blockSize=(block_size[1] * cell_size[1],
                                      block_size[0] * cell_size[0]),
                          _blockStride=(cell_size[1], cell_size[0]),
                          _cellSize=(cell_size[1], cell_size[0]),
                          _nbins=nbins)

  hist = hog.compute(img)
  hist = hist.squeeze()
  return hist
  
 
def sliding_window(img, patch_size= (48,36),
                   istep= 10, jstep = 10, scale=1.0):
    
  Ni, Nj = (int(scale * s) for s in patch_size)
  for i in range(0, img.shape[0] - Ni, istep):
      for j in range(0, img.shape[1] - Ni, jstep):
          patch = img[i:i + Ni, j:j + Nj]
          if scale != 1:
              patch = transform.resize(patch, patch_size)
          yield (i, j), patch

def non_max_suppression(boxes, threshold = 0.5):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > threshold)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")



def detect_faces(model, image, patch_size = (48,36), confidence_threshold = 0.8, overlap_threshold = 0.5):
		# img = color.rgb2gray(image)
	img = image * 255
	positions, patches = zip(*sliding_window(img))
	# patches = np.array(patches)
	# for patch in patches:
	# 	print(patch.shape)
	# 	print(calculate_hog(patch) )
	# 	break
	patches_hog = np.array([calculate_hog(np.array(patch, dtype = np.float32)) for patch in patches])
	# print(patches_hog[1])
	labels = model.predict(patches_hog)
	scores = model.predict_proba(patches_hog)

	positive_indexes = list(np.where(labels == 1)[0])

	positions = np.array(positions)

	boxes = []
	confidences = []
	for positive_index in positive_indexes:
		i, j = positions[positive_index]
		confidence = scores[positive_index][1]
		if confidence >= confidence_threshold:
			x_min, y_min = int(j), int(i)
			x_max, y_max = int(j + patch_size[1]), int(i + patch_size[0])
			box = [x_min,y_min,x_max,y_max]
			boxes.append(box)
			confidences.append(confidence)

	boxes = np.array(boxes)
	confidences = np.array(confidences)
		# final_boxes = non_max_suppression(boxes, threshold = overlap_threshold)
		
		# if visualize:
		# 	for box in final_boxes:
		# 		cv2.rectangle(annotated_img,  (x_min, y_min), (x_max, y_max), (0,0,255), 2)
		# 	return annotated_img
		
	return boxes, confidences

def detect_multiscale(model, img, downscale  = 1.2, patch_size = (48,36), confidence_threshold = 0.9, overlap_threshold = 0.5, visualize = False):
 
  all_boxes = []
  all_confidences = []

  original_width, original_height = img.shape[1], img.shape[0]
  
  image = img.copy()

  # if len(image.shape) < 3:
  # image = color.rgb2gray(image)

  resize_scale = 900/image.shape[1]
  image = cv2.resize(image, (int(image.shape[1] * resize_scale), int(image.shape[0] * resize_scale)))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  

  for im_scaled in pyramid_gaussian(image, downscale=downscale):
    if im_scaled.shape[0] < (patch_size[0]  * 2) or im_scaled.shape[1] < (patch_size[1] * 2):
        break
    # cv2_imshow(im_scaled * 255)
    boxes, confidences = detect_faces(model, im_scaled , patch_size, confidence_threshold , overlap_threshold)
    # print(imsca-)
    for i in range(len(boxes)):
      box = boxes[i]
      confidence = confidences[i]
      x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
      x_min = x_min / im_scaled.shape[1] * original_width
      y_min = y_min / im_scaled.shape[0] * original_height
      x_max = x_max / im_scaled.shape[1] * original_width
      y_max = y_max / im_scaled.shape[0] * original_height
      all_boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
      all_confidences.append(confidence)
        
  

  all_boxes = np.array(all_boxes)
  all_boxes = non_max_suppression(all_boxes, threshold = overlap_threshold)
  all_confidences = np.array(all_confidences)


  if visualize:
    annotated_img = img.copy()
    for box in all_boxes:
      x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
      cv2.rectangle(annotated_img,  (x_min, y_min), (x_max, y_max), (0,0,255), 2)
    return annotated_img
  
  return all_boxes, all_confidences


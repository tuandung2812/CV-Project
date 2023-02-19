import cv2
import numpy as np
import random

import torch

# from utils.box_utils import matrix_iof
import torchvision.transforms.functional as FT
from torchvision import transforms
# random.seed(42)
# np.random.seed(42)

#

def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()
    # brightness disortion
    # _convert(image, beta=random.uniform(-32, 32))
    # Contrast disortion
    # _convert(image, alpha=random.uniform(0.5, 1.5))

    if random.randrange(2):

        #brightness distortion
        if 0 == 0:
            _convert(image, beta=random.uniform(-32, 32))
            # print(image.shape)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def createVerticalKernel(size):
    kernel_size = size

    # Create the kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))

    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)

    # Normalize.
    kernel_v /= kernel_size

    return kernel_v

def createHorizontalKernel(size):
    kernel_size = size
    kernel_h = np.zeros((kernel_size, kernel_size))

    kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel_h /= kernel_size

    return kernel_h


def blur(image):
    blur = image.copy()
    blur = cv2.GaussianBlur(blur, (random.randint(0, 4) * 2 + 1, random.randint(0, 4) * 2 + 1),
                                  cv2.BORDER_DEFAULT)

    if 0 == 0:
        if random.randrange(2):
            blur = cv2.GaussianBlur(blur, (random.randint(0, 4) * 2 + 1, random.randint(0, 4) * 2 + 1),
                                              cv2.BORDER_DEFAULT)
        # Motion blur
        else:
            blur = cv2.filter2D(blur, -1, createVerticalKernel(random.randint(1, 8)))
            blur = cv2.filter2D(blur, -1, createHorizontalKernel(random.randint(1, 8)))

    return blur




def horrizontal_flip(img, bboxes):
    img_center = np.array(img.shape[:2])[::-1] / 2
    img_center = np.hstack((img_center, img_center))
    # if random.randrange(2):
    img = img[:, ::-1, :]
    bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])

    box_w = abs(bboxes[:, 0] - bboxes[:, 2])

    bboxes[:, 0] -= box_w
    bboxes[:, 2] += box_w

    return img, bboxes

def bbox_area(bbox):
    return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])

def clip_box(bbox, clip_box, alpha):
    """Clip the bounding boxes to the borders of an image

    Parameters
    ----------

    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    clip_box: numpy.ndarray
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`

    alpha: float
        If the fraction of a bounding box left in the image after being clipped is
        less than `alpha` the bounding box is dropped.

    Returns
    -------

    numpy.ndarray
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:, 0], clip_box[0]).reshape(-1, 1)
    y_min = np.maximum(bbox[:, 1], clip_box[1]).reshape(-1, 1)
    x_max = np.minimum(bbox[:, 2], clip_box[2]).reshape(-1, 1)
    y_max = np.minimum(bbox[:, 3], clip_box[3]).reshape(-1, 1)

    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))
    
    # print(ar_)
    delta_area = ((ar_ - bbox_area(bbox)) / ar_)

    mask = (delta_area < (1 - alpha)).astype(int)

    bbox = bbox[mask == 1, :]

    return bbox

class RandomScale(object):
    """Randomly scales an image


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    scale: float or tuple(float)
        if **float**, the image is scaled by a factor drawn
        randomly from a range (1 - `scale` , 1 + `scale`). If **tuple**,
        the `scale` is drawn randomly from values specified by the
        tuple

    Returns
    -------

    numpy.ndaaray
        Scaled image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, scale=0.2, diff=False):
        self.scale = scale

        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)

        self.diff = diff

    def __call__(self, img, bboxes):

        # Chose a random digit to scale by
        img_shape = img.shape

        if self.diff:
            scale_x = random.uniform(*self.scale)
            scale_y = random.uniform(*self.scale)
        else:
            scale_x = random.uniform(*self.scale)
            scale_y = scale_x

        resize_scale_x = 1 + scale_x
        resize_scale_y = 1 + scale_y

        img = cv2.resize(img, None, fx=resize_scale_x, fy=resize_scale_y)

        bboxes[:, :4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]

        canvas = np.zeros(img_shape, dtype=np.uint8)

        y_lim = int(min(resize_scale_y, 1) * img_shape[0])
        x_lim = int(min(resize_scale_x, 1) * img_shape[1])

        canvas[:y_lim, :x_lim, :] = img[:y_lim, :x_lim, :]

        img = canvas
        bboxes = clip_box(bboxes, [0, 0, 1 + img_shape[1], img_shape[0]], 0.25)

        return img, bboxes


class RandomTranslate(object):
    """Randomly Translates the image


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    translate: float or tuple(float)
        if **float**, the image is translated by a factor drawn
        randomly from a range (1 - `translate` , 1 + `translate`). If **tuple**,
        `translate` is drawn randomly from values specified by the
        tuple

    Returns
    -------

    numpy.ndaaray
        Translated image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, translate=0.2, diff=False):
        self.translate = translate

        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"
            assert self.translate[0] > 0 & self.translate[0] < 1
            assert self.translate[1] > 0 & self.translate[1] < 1


        else:
            assert self.translate > 0 and self.translate < 1
            self.translate = (-self.translate, self.translate)

        self.diff = diff

    def __call__(self, img, bboxes):
        # Chose a random digit to scale by
        img_shape = img.shape

        # translate the image

        # percentage of the dimension of the image to translate
        translate_factor_x = random.uniform(*self.translate)
        translate_factor_y = random.uniform(*self.translate)

        if not self.diff:
            translate_factor_y = translate_factor_x

        canvas = np.zeros(img_shape).astype(np.uint8)

        corner_x = int(translate_factor_x * img.shape[1])
        corner_y = int(translate_factor_y * img.shape[0])

        # change the origin to the top-left corner of the translated box
        orig_box_cords = [max(0, corner_y), max(corner_x, 0), min(img_shape[0], corner_y + img.shape[0]),
                          min(img_shape[1], corner_x + img.shape[1])]

        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]),
               max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]), :]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3], :] = mask
        img = canvas

        bboxes[:, :4] += [corner_x, corner_y, corner_x, corner_y]

        bboxes = clip_box(bboxes, [0, 0, img_shape[1], img_shape[0]], 0.25)

        return img, bboxes


class RandomShear(object):
    """Randomly shears an image in horizontal direction


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    shear_factor: float or tuple(float)
        if **float**, the image is sheared horizontally by a factor drawn
        randomly from a range (-`shear_factor`, `shear_factor`). If **tuple**,
        the `shear_factor` is drawn randomly from values specified by the
        tuple

    Returns
    -------

    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, shear_factor=0.2):
        self.shear_factor = shear_factor

        if type(self.shear_factor) == tuple:
            assert len(self.shear_factor) == 2, "Invalid range for scaling factor"
        else:
            self.shear_factor = (-self.shear_factor, self.shear_factor)

        shear_factor = random.uniform(*self.shear_factor)

    def __call__(self, img, bboxes):

        shear_factor = random.uniform(*self.shear_factor)

        w, h = img.shape[1], img.shape[0]

        if shear_factor < 0:
            img, bboxes = horrizontal_flip(img, bboxes)

        M = np.array([[1, abs(shear_factor), 0], [0, 1, 0]])

        nW = img.shape[1] + abs(shear_factor * img.shape[0])

        bboxes[:, [0, 2]] += ((bboxes[:, [1, 3]]) * abs(shear_factor)).astype(int)

        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))

        if shear_factor < 0:
            img, bboxes = horrizontal_flip(img, bboxes)

        img = cv2.resize(img, (w, h))

        scale_factor_x = nW / w

        bboxes[:, :4] /= [scale_factor_x, 1, scale_factor_x, 1]

        return img, bboxes


def _resize_subtract_mean(image, insize):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    # cv2.imshow("Resized",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image = cv2.resize(image, insize, interpolation=interp_method)
    # image = image.astype(np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # image = FT.to_tensor(image)
    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    # to_tensor = transforms.ToTensor()
    # image = to_tensor(image)
    # # print(image)
    # image = FT.normalize(image,mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
    # image = transform(image)
    # image = FT.normalize(image, mean=mean, std=std)
    # print(image)
    # image = (image - mean)/std

    # image -= np.mean(image, axis=0)
    # image /= np.std(image, axis = 0)
    return image

def _resize(image, insize):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    # cv2.imshow("Resized",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image = cv2.resize(image, insize, interpolation=interp_method)
    transform = transforms.Compose([transforms.ToTensor()
    ])
    image = transform(image)
    return image

class preproc(object):

    def __init__(self, img_dim, augment = True):
        self.img_dim = img_dim
        self.augment = augment


    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes = targets[:, :4].copy()
        labels = targets[:, -1].copy()
        # landm = targets[:, 4:-1].copy()
        original_width  = image.shape[1]
        original_height = image.shape[0]

        # print(landm, labels)
        # image_resized = cv2.resize(image,self.img_dim)
        # image_resized = image_resized/ 255.0
        # # print(image_width, image_height, self.img_dim[0], self.img_dim[1])
        # scale_x = self.img_dim[1] / original_width
        # scale_y = self.img_dim[0] / original_height
        # # print(targets)
        # new_boxes = []
        # for box in boxes:
        #     xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        #     xmin_final = xmin * scale_x
        #     xmax_final = xmax * scale_x
        #     ymin_final = ymin * scale_y
        #     ymax_final = ymax * scale_y
        #
        #     new_boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

            # print(box)
        if self.augment:
            image_t = image
            image_t = _distort(image_t)
            boxes_t = boxes
            if random.randrange(2):
                image_t = blur(image_t)
                # boxes_t =boxes
                image_t, boxes_t = horrizontal_flip(image_t, boxes)
            if random.randrange(2):
                scale_transform = RandomScale()
                image_t, boxes_t = scale_transform(image_t, boxes_t)

            if random.randrange(2):
                translate_transform = RandomTranslate(translate = 0.1)
                image_t, boxes_t = translate_transform(image_t, boxes_t)
            if random.randrange(2):
                shear_transform = RandomShear()
                image_t, boxes_t = shear_transform(image_t, boxes_t)
        else:
            boxes_t = boxes
            image_t = image

        height, width, _ = image_t.shape
        image_t = _resize_subtract_mean(image_t, self.img_dim)

        scale_x = self.img_dim[1] / original_width
        scale_y = self.img_dim[0] / original_height

        new_boxes = []
        for box in boxes_t:
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
            xmin_final = (xmin * scale_x)/self.img_dim[1]
            xmax_final = (xmax * scale_x)/self.img_dim[1]
            ymin_final = (ymin * scale_y)/self.img_dim[0]
            ymax_final = (ymax * scale_y)/self.img_dim[0]

            new_boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])



        # image_t, boxes_t, labels_t, landm_t, pad_image_flag = _crop(image, boxes, labels, self.img_dim)
        # image_t = _distort(image_t)
        # image_t = _pad_to_square(image_t,self.rgb_means, pad_image_flag)
        # height, width, _ = image_t.shape
        # image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)
        # cv2.imshow(image_t)
        # cv2.waitKey(0)
        # boxes_t[:, 0::2] /= width
        # boxes_t[:, 1::2] /= height
        #
        # landm_t[:, 0::2] /= width
        # landm_t[:, 1::2] /= height
        #
        # labels_t = np.expand_dims(labels_t, 1)
        # targets_t = np.hstack((boxes_t, landm_t, labels_t))
        #
        return image_t, new_boxes, labels


class preproc_faster_rcnn(object):

    def __init__(self, img_dim, augment = True):
        self.img_dim = img_dim
        self.augment = augment


    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes = targets[:, :4].copy()
        labels = targets[:, -1].copy()
        # landm = targets[:, 4:-1].copy()
        original_width  = image.shape[1]
        original_height = image.shape[0]
        # print(image.shape)

        # print(landm, labels)
        # image_resized = cv2.resize(image,self.img_dim)
        # image_resized = image_resized/ 255.0
        # # print(image_width, image_height, self.img_dim[0], self.img_dim[1])
        # scale_x = self.img_dim[1] / original_width
        # scale_y = self.img_dim[0] / original_height
        # # print(targets)
        # new_boxes = []
        # for box in boxes:
        #     xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        #     xmin_final = xmin * scale_x
        #     xmax_final = xmax * scale_x
        #     ymin_final = ymin * scale_y
        #     ymax_final = ymax * scale_y
        #
        #     new_boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

            # print(box)
        if self.augment:
            image_t = image
            image_t = _distort(image_t)
            boxes_t = boxes
            if random.randrange(2):
                image_t = blur(image_t)
                # boxes_t =boxes
                image_t, boxes_t = horrizontal_flip(image_t, boxes)
            if random.randrange(2):
                scale_transform = RandomScale()
                image_t, boxes_t = scale_transform(image_t, boxes_t)

            if random.randrange(2):
                translate_transform = RandomTranslate(translate = 0.1)
                image_t, boxes_t = translate_transform(image_t, boxes_t)
            if random.randrange(2):
                shear_transform = RandomShear()
                image_t, boxes_t = shear_transform(image_t, boxes_t)
        else:
            boxes_t = boxes
            image_t = image

        height, width, _ = image_t.shape
        image_t = _resize(image_t, self.img_dim)
        # image_t /= 255.0

        scale_x = self.img_dim[1] / original_width
        scale_y = self.img_dim[0] / original_height

        new_boxes = []
        for box in boxes_t:
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
            xmin_final = (xmin/original_width)*self.img_dim[1]
            xmax_final = (xmax /original_width)*self.img_dim[1]
            ymin_final = (ymin /original_height)*self.img_dim[0]
            ymax_final = (ymax /original_height)*self.img_dim[0]

            new_boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        boxes = torch.as_tensor(new_boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels




        # image_t, boxes_t, labels_t, landm_t, pad_image_flag = _crop(image, boxes, labels, self.img_dim)
        # image_t = _distort(image_t)
        # image_t = _pad_to_square(image_t,self.rgb_means, pad_image_flag)
        # height, width, _ = image_t.shape
        # image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)
        # cv2.imshow(image_t)
        # cv2.waitKey(0)
        # boxes_t[:, 0::2] /= width
        # boxes_t[:, 1::2] /= height
        #
        # landm_t[:, 0::2] /= width
        # landm_t[:, 1::2] /= height
        #
        # labels_t = np.expand_dims(labels_t, 1)
        # targets_t = np.hstack((boxes_t, landm_t, labels_t))
        #
        return image_t, target

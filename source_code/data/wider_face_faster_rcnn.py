import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from .data_augment import preproc, preproc_faster_rcnn

import torchvision.transforms.functional as FT
from torchvision import transforms

class WiderFaceFasterRCNN(data.Dataset):
    def __init__(self, txt_path, img_dim, augment = True):
        self.img_dim = img_dim
        self.augment = augment

        self.preproc = preproc_faster_rcnn(img_dim = self.img_dim,augment = self.augment)
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        # print(lines[0:5])
        isFirst = True
        labels = []
        for i in range(len(lines)):
            line = lines[i]
            line = line.rstrip()

        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):

            # print(line)
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
#                 path = txt_path.replace('gt.txt','images/') + path
                if "train_gt.txt" in txt_path:
                    path = txt_path.replace('train_gt.txt', 'images/') + path
                elif "val_gt.txt" in txt_path:
                    path = txt_path.replace('val_gt.txt', 'images/') + path
                elif "train_augment_gt.txt" in txt_path:
                    path = txt_path.replace('train_augment_gt.txt', 'images/') + path
                # print(path)
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                # print(line)
                bboxes = [float(x) for x in line]
                labels.append(bboxes)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        # print(self.imgs_path[index])
        img = cv2.imread(self.imgs_path[index])
        # print(self.imgs_path[index])
        # convert BGR to RGB color format
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        height, width, _ = img.shape
        labels = self.words[index]
        # print(labels)

        annotations = np.zeros((0, 5))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 5))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            # annotation[0, 4] = label[4]    # l0_x
            # annotation[0, 5] = label[5]    # l0_y
            # annotation[0, 6] = label[7]    # l1_x
            # annotation[0, 7] = label[8]    # l1_y
            # annotation[0, 8] = label[10]   # l2_x
            # annotation[0, 9] = label[11]   # l2_y
            # annotation[0, 10] = label[13]  # l3_x
            # annotation[0, 11] = label[14]  # l3_y
            # annotation[0, 12] = label[16]  # l4_x
            # annotation[0, 13] = label[17]  # l4_y
            # if (annotation[0, 4]<0):
            #     annotation[0, 14] = -1
            # else:
            annotation[0, 4] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        img, target = self.preproc(img, target)

        # print(img)
        return img, target


def detection_collate(batch):
    return tuple(zip(*batch))

if __name__ == '__main__':
    import os
    import os.path
    import sys
    import torch
    import torch.utils.data as data
    import cv2
    import numpy as np
    from data_augment import preproc

    # data_path = "./dataset/wider_face_split/wider_face_train_bbx_gt.txt"
    data_path = '../dataset/wider_face/train/train_gt.txt'
    # rgb_mean = (104, 117, 123)  # bgr order
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img_size = (300, 300)
    dataset = WiderFaceFasterRCNN(txt_path=data_path,img_dim=img_size, augment=False)
    # print(dataset[200])
    x, target = dataset[100]
    y, label = target['boxes'], target['labels']
    print(x,y)
    # invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
    #                                                     std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    #                                transforms.Normalize(mean=[-0.485, -0.456, -0.406],
    #                                                     std=[1., 1., 1.]),
    #                                ])
    # x = invTrans(x)

    # print(labels)
    x = x.cpu().detach().numpy()
    x = x.transpose(1,2,0)
    # print(x.shape, x)
    img = x.copy()
    for box in y:
        bbox = box[0:4]
        # print(bbox)
        cv2.rectangle(
            img,
            (int(bbox[0] * 300), int(bbox[1] * 300)), (int(bbox[2] * 300), int(bbox[3] * 300)),
            (0, 255, 0), 2
        )
        # print(label)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False,
                                               collate_fn=detection_collate, num_workers=4,
                                               pin_memory=True)  # note that we're passing the collate function here
    print(len(dataset))
    # for _,batch in enumerate()
    # for i, (images, boxes, labels) in enumerate(train_loader):
    #     print(images[0], boxes[0], labels[0])
    #     break

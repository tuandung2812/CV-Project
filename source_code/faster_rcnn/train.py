import torch
import torchvision
import os.path
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from faster_rcnn.model import get_object_detection_model
from utils import *
from data.wider_face_faster_rcnn import WiderFaceFasterRCNN, detection_collate
import yaml
import datetime
from tqdm.notebook import tqdm

print_freq = 300

def validate(val_loader, model, device):
  eval_losses = []
  # model.eval()
  i = 0
  with torch.no_grad():
    for i, (imgs,annotations) in enumerate(val_loader):
        try:
          # print(i)
          # Move to default device
          i += 1
          imgs = list(img.to(device) for img in imgs)
          # for t in annotations:
          #   for k, v in t.items():
          #     print(k, v.shape)
          annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

          with torch.no_grad():
            loss_dict = model(imgs, annotations)
            # print(loss_dict)

          # loss_dict = model(imgs, annotations) 
          # print(type(loss_dict))
          losses = sum(loss for loss in loss_dict.values())        

          if losses < 50:
            eval_losses.append(losses)

        except:
            pass
        #   print(i)
  mean_eval_loss = sum(eval_losses) / len(eval_losses)

  return mean_eval_loss

def train_faster_rcnn(config,device, augment = True, pretrained = True):
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at
    print(device)

    n_classes = config['n_classes']
    checkpoint = config['checkpoint']
    batch_size = config['batch_size']
    iterations = config['iterations']
    workers = config['workers']
    print_freq = config['print_freq']
    lr = float(config['lr'])
    momentum = float(config['momentum'])
    print(checkpoint)
    # decay_lr_at = config['decay_lr_at']
    decay_lr_to = float(config['decay_lr_to'])
    # momentum = float(config['momentum'])
    weight_decay = float(config['weight_decay'])
    # print(checkpoint)
    # grad_clip = 1

    # Initialize model or load checkpoint
    if checkpoint is None or not os.path.isfile(checkpoint):
        start_epoch = 0
        model =  get_object_detection_model(num_classes = n_classes, from_pretrained =  pretrained)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 3)
        best_loss = 999
    else:
        print("Loaded previous checkpoint")
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model =  get_object_detection_model(num_classes = n_classes, from_pretrained =  pretrained)
        model.load_state_dict(checkpoint['model'])
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

    # Move to default device
    model = model.to(device)
    # print(model)
    
    img_size = (300, 300)
    grad_clip = 2  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

    cudnn.benchmark = True
    print_freq = 500
    train_data_path = '/content/data/wider_face/train/train_gt.txt'

    train_augment_path = '/content/data/wider_face/train/train_gt.txt'
    val_path  = '/content/data/wider_face/train/val_gt.txt'



    train_dataset = WiderFaceFasterRCNN(txt_path=train_data_path,img_dim=img_size, augment=False)
    
    if augment:
      train_augment_dataset =  WiderFaceFasterRCNN(txt_path=train_data_path,img_dim=img_size, augment=True)
      train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_augment_dataset])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=detection_collate, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    val_dataset =  WiderFaceFasterRCNN(txt_path=val_path,img_dim=img_size, augment=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                               collate_fn=detection_collate, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = iterations // (len(train_dataset) // batch_size)
    # decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]

    current_best_loss = best_loss
    # print(iterations, len(train_dataset) // batch_size)
    # Epochs
    for epoch in range(start_epoch, epochs):
        # print(epoch)
        start_time = datetime.datetime.now()

        print("********* Epoch {0}/{1}  *********".format(epoch, epochs))

        # Decay learning rate at particular epochs
        # if epoch in decay_lr_at:
        #     adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        current_best_loss = train(train_loader=train_loader,
              val_loader = val_loader,
              model=model,
              checkpoint = checkpoint,
              optimizer=optimizer,
              scheduler = scheduler,
              epoch=epoch,
              best_loss = current_best_loss,
              device = device)
      
        current_time = datetime.datetime.now()
        time_taken = current_time - start_time
        print("Time taken for epoch: ", time_taken)

    #     # print(loss)
    #     # with open('loss.txt', 'a') as f:
    #     #     f.write(loss + '\n')
    #     #
    #     # # Save checkpoint
    #     # save_checkpoint(checkpoint,epoch, model, optimizer)


def train(train_loader,val_loader, model, checkpoint, optimizer, scheduler, epoch, best_loss, device):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    # losses = AverageMeter()  # loss

    start = time.time()

    epoch_losses = []

    current_best_loss = best_loss
    # Batches
    i = 0
    for imgs, annotations in tqdm(train_loader):
        # print(i)
        data_time.update(time.time() - start)
        # Move to default device
        imgs = list(img.to(device) for img in imgs)
        # for t in annotations:
        #   for k, v in t.items():
        #     print(k, v.shape)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        # print(predicted_locs)

        # Loss
        try:
          loss_dict = model(imgs, annotations) 
          # print(loss_dict)
          losses = sum(loss for loss in loss_dict.values())        

          optimizer.zero_grad()
          losses.backward()

          # # Clip gradients, if necessary
          # if grad_clip is not None:
          clip_gradient(optimizer, grad_clip)

          # Update model
          optimizer.step()

          # losses.update(loss.item(), images.size(0))
          batch_time.update(time.time() - start)

          start = time.time()

          epoch_losses.append(losses)

        except:
          pass
          # print(i)

        # Backward prop.
        # Print status
        if i % print_freq == 0:
            # loss = sum(epoch_losses) / len(epoch_losses)
            eval_loss = validate( val_loader, model, device)
            scheduler.step(eval_loss)
            print(best_loss)

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss:.4f}\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=eval_loss))

            with open('loss.txt', 'a') as f:
                f.write(str(eval_loss) + '\n')
            if eval_loss < current_best_loss:
                # Save checkpoint
                best_loss = eval_loss
                current_best_loss = eval_loss
                save_checkpoint(checkpoint,epoch = epoch, model = model,  optimizer = optimizer, scheduler = scheduler, best_loss = current_best_loss)
                print("Model saved")
        i += 1


    del imgs,annotations  # free some memory since their histories may be stored
    return current_best_loss

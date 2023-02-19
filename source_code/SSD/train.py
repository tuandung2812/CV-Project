import os.path
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from SSD.model import SSD300, MultiBoxLoss
from utils import *
from data.wider_face import WiderFaceDetection, detection_collate
import yaml
import datetime

# Data parameters
train_data_path = '/content/data/wider_face/train_gt.txt'

train_augment_path = '/content/data/wider_face/train_augment_gt.txt'
val_path  = '/content/data/wider_face/val_gt.txt'
# keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_size = (300,300)
train_dataset = WiderFaceDetection(txt_path=train_data_path,img_dim=img_size, augment=False)
train_augment_dataset = WiderFaceDetection(txt_path=train_augment_path,img_dim=img_size, augment=True)
val_dataset = WiderFaceDetection(txt_path=val_path,img_dim=img_size, augment=False)

grad_clip = 2  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True
print_freq = 500

def validate(val_loader, model,criterion, device):
  eval_losses = []
  model.eval()
  with torch.no_grad():
    for i, (images, boxes, labels) in enumerate(val_loader):
        # print(i)
        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # print(predicted_locs)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
        # print(loss)
        if loss.item() < 50:
          eval_losses.append(loss.item())
  mean_eval_loss = sum(eval_losses) / len(eval_losses)

  return mean_eval_loss

def train_ssd(config,device,augment = True):
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
    decay_lr_at = config['decay_lr_at']
    decay_lr_to = float(config['decay_lr_to'])
    # momentum = float(config['momentum'])
    weight_decay = float(config['weight_decay'])
    # print(checkpoint)
    # grad_clip = 1

    # Initialize model or load checkpoint
    if checkpoint is None or not os.path.isfile(checkpoint):
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        # optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
        #                             lr=lr, momentum=momentum, weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                            lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 3)

        best_loss = 999
    else:
        print("Loaded previous checkpoint")
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = SSD300(n_classes=n_classes)
        # print(checkpoint['model'])
        model.load_state_dict(checkpoint['model'])
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    # print(model)
    
    img_size = (300, 300)

    train_dataset = WiderFaceDetection(txt_path=train_data_path,img_dim=img_size, augment=False)
    if augment:
        train_augment_dataset =  WiderFaceDetection(txt_path=train_data_path,img_dim=img_size, augment= True)
      
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_augment_dataset])
    

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=detection_collate, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    val_dataset =  WiderFaceDetection(txt_path=val_path,img_dim=img_size, augment=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                               collate_fn=detection_collate, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = iterations // (len(train_dataset) // batch_size)
    decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]

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
              criterion=criterion,
              optimizer=optimizer,
              scheduler = scheduler,
              epoch=epoch,
              best_loss = current_best_loss)
      
        current_time = datetime.datetime.now()
        time_taken = current_time - start_time
        print("Time taken for epoch: ", time_taken)

    #     # print(loss)
    #     # with open('loss.txt', 'a') as f:
    #     #     f.write(loss + '\n')
    #     #
    #     # # Save checkpoint
    #     # save_checkpoint(checkpoint,epoch, model, optimizer)


def train(train_loader,val_loader, model, checkpoint, criterion, optimizer, scheduler, epoch, best_loss):
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
    for i, (images, boxes, labels) in enumerate(train_loader):
        # print(i)
        data_time.update(time.time() - start)
        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # print(predicted_locs)

        # Loss
        try:
          loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
          optimizer.zero_grad()
          loss.backward()

          # # Clip gradients, if necessary
          # if grad_clip is not None:
          clip_gradient(optimizer, grad_clip)

          # Update model
          optimizer.step()

          # losses.update(loss.item(), images.size(0))
          batch_time.update(time.time() - start)

          start = time.time()

          epoch_losses.append(loss.item())

        except:
          pass
          # print(i)

        # Backward prop.
        # Print status
        if i % print_freq == 0:
            # loss = sum(epoch_losses) / len(epoch_losses)
            eval_loss = validate( val_loader, model, criterion, device)
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

    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored
    return current_best_loss
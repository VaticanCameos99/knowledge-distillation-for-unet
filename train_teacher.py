import glob
import torch
import dataset
import numpy as np
from unet import UNet
import torch.nn as nn
from metrics import dice_loss
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

num_of_epochs = 5

def evaluate(teacher, val_loader):
    teacher.eval().cuda()

    criterion = nn.BCEWithLogitsLoss()
    ll = []
    with torch.no_grad():
        for i,(img,gt) in enumerate(val_loader):
            img = img[0, :, :, :, :]
            gt = gt[0,:, :, :, :]
            if torch.cuda.is_available():
                img, gt = img.cuda(), gt.cuda()
            img, gt = Variable(img), Variable(gt)

            output = teacher(img)
            loss = criterion(output, gt)
            ll.append(loss.item())

    
    mean_dice = np.mean(ll)
    print('Eval metrics:\n\tAverabe Dice loss:{}'.format(mean_dice))


def train(teacher, optimizer, train_loader):
    print(' --- teacher training')
    teacher.train().cuda()
    criterion = nn.BCEWithLogitsLoss()
    ll = []
    for i, (img, gt) in enumerate(train_loader):
        img = img[0, :, :, :, :]
        gt = gt[0,:, :, :, :]
        if torch.cuda.is_available():
            img, gt = img.cuda(), gt.cuda()
        
        img, gt = Variable(img), Variable(gt)
        output = teacher(img)
        loss = criterion(output, gt)
        ll.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    
    mean_dice = np.mean(ll)

    print("Average loss over this epoch:\n\tDice:{}".format(mean_dice))

if __name__ == "__main__":

    teacher = UNet(channel_depth = 32, n_channels = 3, n_classes=1)

    optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size = 100, gamma = 0.2)

    #load teacher and student model

    #NV: add val folder
    train_list = glob.glob('/home/nirvi/NIO/diabetic-20200318T171807Z-001/diabetic/images/*.jpg')
    val_list = glob.glob('/home/nirvi/NIO/diabetic-20200318T171807Z-001/diabetic/test_images/*.jpg')

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #2 tensors -> img_list and gt_list. for batch_size = 1 --> img: (1, 3, 320, 320); gt: (1, 1, 320, 320)
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
        shuffle = False,
        transform = tf,
        ),
        batch_size = 1
    )


    val_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
        shuffle = False,
        transform = tf,
        ),
        batch_size = 1
    )    

    for epoch in range(num_of_epochs):
        print(' --- teacher training: epoch {}'.format(epoch+1))
        train(teacher, optimizer, train_loader)

        #evaluate for one epoch on validation set
        evaluate(teacher, val_loader)

        #if val_metric is best, add checkpoint

        torch.save(teacher.state_dict(), 'checkpoints_bce/CP{}.pth'.format(epoch+1))
        print("Checkpoint {} saved!".format(epoch+1))
        scheduler.step()
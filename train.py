import glob
import torch
import dataset
import numpy as np
from utils import *
from unet import UNet
from loss import loss_fn_kd
from metrics import dice_loss
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR


teacher_weights = '/home/nirvi/Internship_2020/KDforUNET/teacher_checkpoints/32_final/CP_32_5.pth'
#student_weights = 'checkpoints/CP5.pth'
num_of_epochs = 5
summary_steps = 10

def fetch_teacher_outputs(teacher, train_loader):
    print('-------Fetch teacher outputs-------')
    teacher.eval().cuda()
    #list of tensors
    teacher_outputs = []
    with torch.no_grad():
        #trainloader gets bs images at a time. why does enumerate(tl) run for all images?
        for i, (img, gt) in enumerate(train_loader):
            print(i, 'i')
            '''img = img[0, :, :, :, :]
            gt = gt[0, :, :, :, :]'''
            if torch.cuda.is_available():
                img = img.cuda(async = True)
            img = Variable(img)

            output = teacher(img)
            teacher_outputs.append(output)
    return teacher_outputs

def train_student(student, teacher_outputs, optimizer, train_loader):
    print('-------Train student-------')
    #called once for each epoch
    student.train().cuda()

    summ = []
    for i, (img, gt) in enumerate(train_loader):
        teacher_output = teacher_outputs[i]
        if torch.cuda.is_available():
            img, gt = img.cuda(), gt.cuda()
            teacher_output = teacher_output.cuda()

        img, gt = Variable(img), Variable(gt)
        teacher_output =  Variable(teacher_output)

        output = student(img)

        #TODO: loss is wrong
        loss = loss_fn_kd(output, teacher_output, gt)    

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()
        if i % summary_steps == 0:
            #do i need to move it to CPU?
            
            metric = dice_loss(output, gt)
            summary = {'metric' : metric.item(), 'loss' : loss.item()}
            summ.append(summary)
    
    #print('Average loss over this epoch: ' + np.mean(loss_avg))
    mean_dice_coeff =  np.mean([x['metric'] for x in summ])
    mean_loss = np.mean([x['loss'] for x in summ])
    print('- Train metrics:\n' + '\tMetric:{}\n\tLoss:{}'.format(mean_dice_coeff, mean_loss))
    #print accuracy and loss

def evaluate_kd(student, val_loader):
    print('-------Evaluate student-------')
    student.eval().cuda()

    #criterion = torch.nn.BCEWithLogitsLoss()
    loss_summ = []
    with torch.no_grad():
        for i, (img, gt) in enumerate(val_loader):
            if torch.cuda.is_available():
                img, gt = img.cuda(), gt.cuda()
            img, gt = Variable(img), Variable(gt)

            output = student(img)
            output = output.clamp(min = 0, max = 1)
            loss = dice_loss(output, gt)

            loss_summ.append(loss.item())

    mean_loss = np.mean(loss_summ)
    print('- Eval metrics:\n\tAverage Dice loss:{}'.format(mean_loss))
    return mean_loss

if __name__ == "__main__":
    min_loss = 100

    teacher = UNet(channel_depth = 32, n_channels = 3, n_classes=1)
    student = UNet(channel_depth = 16, n_channels = 3, n_classes=1)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size = 100, gamma = 0.2)

    #load teacher and student model
    teacher.load_state_dict(torch.load(teacher_weights))
    #student.load_state_dict(torch.load(student_weights))

    #NV: add val folder
    train_list = glob.glob('/home/nirvi/Internship_2020/Carvana dataset/train/train1/*jpg')
    val_list = glob.glob('/home/nirvi/Internship_2020/Carvana dataset/val/val1/*jpg')

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

    #train_and_evaluate_kd:
    #get teacher outputs as list of tensors
    teacher_outputs = fetch_teacher_outputs(teacher, train_loader)
    print(len(teacher_outputs))
    for epoch in range(num_of_epochs):
        #train the student
        print(' --- student training: epoch {}'.format(epoch+1))
        train_student(student, teacher_outputs, optimizer, train_loader)

        #evaluate for one epoch on validation set
        val = evaluate_kd(student, val_loader)
        if(val < min_loss):
            min_loss = val
            #TODO: make min as the val loss of teacher
            print('New best!!')


        #if val_metric is best, add checkpoint

        torch.save(student.state_dict(), 'checkpoints/0.9/16/CP{}.pth'.format(epoch+1))
        print("Checkpoint {} saved!".format(epoch+1))
        scheduler.step()
        







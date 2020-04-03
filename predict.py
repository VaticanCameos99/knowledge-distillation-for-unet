import torch
from unet import UNet
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable

student_weights = 'MODEL.pth'

tf1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

tf2 = transforms.Compose([
    transforms.ToPILImage()
])

def plot_img_and_output(img, output):
    fig = plt.figure()

    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input')
    plt.imshow(img)

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output')
    plt.imshow(output)

    plt.show()

def predict_output(student, img):
    student.eval()
    with torch.no_grad():
        img = tf1(img)
        img = img.float().cuda().unsqueeze(0)
        
        output = student(img)
        output = output.clamp(min = 0, max = 1)
        return output


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images', required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = getargs()
    input_file = args.input
    output_file = args.output

    print(input_file)
    print(output_file)

    student = UNet(channel_depth = 64, n_channels = 3, n_classes=1)
    student.eval().cuda()
    student.load_state_dict(torch.load(student_weights))
    print('Model weights loaded!')

    for i, fn in enumerate(input_file):
        print("\nPredicting image {} ...".format(fn))
        img = Image.open(fn).resize((640, 959))
        output = predict_output(student, img)
        output = output.squeeze(0)
        print(img.size, 'img.shape')
        output = tf2(output.cpu())
        print(output.size, 'output')
        plot_img_and_output(img, output)
        #img.save('input2.jpg')
        output.save(output_file[i])
        exit(1)




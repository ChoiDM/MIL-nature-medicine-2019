import os
import cv2
import sys
import random
import argparse
import openslide
import numpy as np
import pandas as pd
from glob import glob
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models


parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--dataroot', type=str, default='/data/crc_orig/paip2020_new/patch/MSI_classification_patch_level1', help='abnormal patch directory')
parser.add_argument('--output', type=str, default='.', help='name of output file')
parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size (default: 512)')
parser.add_argument('--input_size', type=int, default=512, help='image input size (default: 224)')
parser.add_argument('--resume', type=str, default='', help='path to pretrained model')
parser.add_argument('--workers', default=10, type=int, help='number of data loading workers (default: 4)')

def main():
    global args
    args = parser.parse_args()

    #cnn
    model = models.resnet34(True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.cuda()

    if args.resume:
        ch = torch.load(args.resume)
        model.load_state_dict(ch['state_dict'])
    
    trans = None

    #load data
    train_dset = MILProbdataset(args, transform=trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    #open output file
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    #loop throuh epochs
    probs, path_list = inference(train_loader, model)
    
    prob_df = pd.DataFrame([path_list, probs]).T
    prob_df.columns = ['path', 'probability']
    prob_df.to_csv('calculated_probs.csv', index=False)


def inference(loader, model):
    model.eval()

    probs = torch.FloatTensor(len(loader.dataset))
    paths = []

    with torch.no_grad():
        for i, (input, path) in enumerate(loader):
            if (i+1) % 10==0:
                print('Inference\tBatch: [{}/{}]'.format(i+1, len(loader)))
                # break
            input = input.cuda().float()
            output = model(input)

            output = F.softmax(output, dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
            paths.extend(list(np.array(path).reshape(-1)))
    return probs.cpu().numpy(), paths


class MILProbdataset(data.Dataset):
    def __init__(self, args, transform=None):
        self.dataroot = args.dataroot

        #Flatten grid
        patch_paths = []
        slideIDX = []
        # targets = []

        img_list = sorted(glob(os.path.join(args.dataroot, '*', '*', '*.png')))
        for i, patch_path in enumerate(img_list):
            patch_paths.append(patch_path)
            slideIDX.append(self.get_slide_idx_from_path(patch_path))
            # targets.append(int(patch_path.split(os.sep)[-2]))

        print('Number of patches: {}'.format(len(patch_paths)))

        self.slideIDX = slideIDX
        # self.targets = targets
        self.patch_paths = patch_paths

        self.transform = transform

        self.size = args.input_size
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def get_slide_idx_from_path(self, path):
        filename = os.path.basename(path)
        return int(filename.split('_')[2])

    def __getitem__(self,index):
        patch_path = self.patch_paths[index]

        img = self.read_img(patch_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, patch_path


    def __len__(self):
        return len(self.patch_paths)

    def read_img(self, path):
        img = cv2.imread(path)
        if self.size != 512:
            img = cv2.resize(img, (self.size, self.size))
        img = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = np.moveaxis(img, -1, 0)

        return img

if __name__ == '__main__':
    main()
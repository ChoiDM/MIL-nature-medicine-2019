import sys
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import random
import openslide
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

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataroot', type=str, default='/data/crc_orig/paip2020_new/patch/test_MSI_CNN_input_patch', help='abnormal patch directory')
parser.add_argument('--output', type=str, default='.', help='name of output directory')
parser.add_argument('--input_size', type=int, default=512, help='image input size (default: 224)')
parser.add_argument('--model', type=str, default='', help='path to pretrained model')
parser.add_argument('--batch_size', type=int, default=256, help='how many images to sample per slide (default: 100)')
parser.add_argument('--workers', default=10, type=int, help='number of data loading workers (default: 4)')

def main():
    global args
    args = parser.parse_args()

    #load model
    model = models.resnet34(True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    ch = torch.load(args.model)
    model.load_state_dict(ch['state_dict'])
    model = model.cuda()
    cudnn.benchmark = True

    #normalization
    # normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    # trans = transforms.Compose([transforms.ToTensor(),normalize])
    trans = None

    #load data
    dset = MILdataset(args, is_Train=False, transform=trans)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    dset.setmode(1)
    probs = inference(loader, model)
    slide_idcs, max_probs = group_max(np.array(dset.slideIDX), probs)

    fp = open(os.path.join(args.output, 'predictions.csv'), 'w')
    fp.write('file,prediction,probability\n')
    for name, prob in zip(slide_idcs, max_probs):
        fp.write('{},{},{}\n'.format(name, int(prob>=0.5), prob))
    fp.close()

def inference(loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        i = 0
        for input in tqdm(loader):
            # print('Batch: [{}/{}]'.format(i+1, len(loader)))
            input = input.cuda().float()
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
            i += 1
    return probs.cpu().numpy()

def group_max(groups, data):
    nmax = len(data)
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    out = out[~np.isnan(out)]
    # out_bi = (out > 0.5).astype(np.uint8)
    return groups[index], out

class MILdataset(data.Dataset):
    def __init__(self, args, is_Train, transform=None):
        self.dataroot = args.dataroot
        self.is_Train = is_Train

        #Flatten grid
        patch_paths = []
        slideIDX = []

        img_list = sorted(glob(os.path.join(args.dataroot, '*', 'test', '*.png')))
        for i, patch_path in enumerate(img_list):
            patch_paths.append(patch_path)
            slideIDX.append(self.get_slide_idx_from_path(patch_path))

        print('Number of patches: {}'.format(len(patch_paths)))

        self.slideIDX = slideIDX
        self.patch_paths = patch_paths

        self.transform = transform
        self.mode = None

        self.size = args.input_size
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def get_slide_idx_from_path(self, path):
        filename = os.path.basename(path)
        return int(filename.split('_')[2])

    def setmode(self,mode):
        self.mode = mode

    def __getitem__(self,index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            patch_path = self.patch_paths[index]

            img = self.read_img(patch_path)
            if self.transform is not None:
                img = self.transform(img)
            return img

        elif self.mode == 2:
            slideIDX, patch_path, target = self.t_data[index]

            img = self.read_img(patch_path)
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode == 1:
            return len(self.patch_paths)
        elif self.mode == 2:
            return len(self.t_data)

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

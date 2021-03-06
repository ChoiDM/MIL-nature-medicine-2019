import os
import cv2
import sys
import random
import argparse
import openslide
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models

##############################################################
# | Top-k |  LR  | class weights | batch_size | input_size | #
# |-------|------|---------------|------------|------------| #
# |   1   | 1e-4 |   0.25:0.75   |     64     |   512x512  | #
# |   3   | 1e-4 |   0.25:0.75   |     64     |   512x512  | #
# |   5   | 1e-4 |   0.25:0.75   |     32     |   512x512  | #
# |   7   | 1e-4 |   0.25:0.75   |     16     |   512x512  | #
##############################################################

parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--train_dir', type=str, default='/data/crc_orig/paip2020_new/patch/MSI_classification_patch_level1', help='abnormal patch directory')
parser.add_argument('--valid_dir', type=str, default='/data/crc_orig/paip2020_new/patch/val_MSI_CNN_input_patch', help='abnormal patch directory')
parser.add_argument('--valid_annot', type=str, default='/data/dmchoi/validation_gt.csv', help='GT annotation of validation dataset')
parser.add_argument('--overlap', action='store_true', help='use this option to overlap patches')
parser.add_argument('--output', type=str, default='.', help='name of output file')
parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size (default: 512)')
parser.add_argument('--input_size', type=int, default=512, help='image input size (default: 224)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--resume', type=str, default='', help='path to pretrained model')
parser.add_argument('--workers', default=10, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=5, type=int, help='test on val every (default: 10)')
parser.add_argument('--weights', default=0.75, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--k', default=1, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')

best_loss = 1e+10
def main():
    global args, best_loss
    args = parser.parse_args()

    #cnn
    model = models.resnet34(True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.cuda()

    if args.resume:
        ch = torch.load(args.resume)
        model.load_state_dict(ch['state_dict'])
    

    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights,args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    cudnn.benchmark = True

    #normalization
    # normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    # trans = transforms.Compose([transforms.ToTensor(), normalize])
    trans = None

    #load data
    train_dset = MILTraindataset(args, transform=trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
        
    val_dset = MILValiddataset(args, transform=trans)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    #open output file
    args.output = os.path.join(args.output, 'weights_%s_lr_%s_in_%d_top%d'%(args.weights, args.lr, args.input_size, args.k))
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    fconv = open(os.path.join(args.output,'convergence.csv'), 'w')
    fconv.write('epoch,metric,value\n')
    fconv.close()

    #loop throuh epochs
    for epoch in range(args.nepochs):
        print("\nTraining...")
        train_dset.setmode(1)
        probs, _ = inference(epoch, train_loader, model, criterion)
        topk = group_argtopk(np.array(train_dset.slideIDX), probs, args.k)
        train_dset.maketraindata(topk)
        train_dset.shuffletraindata()
        train_dset.setmode(2)
        loss = train(epoch, train_loader, model, criterion, optimizer)
        print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch+1, args.nepochs, loss))
        fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch+1,loss))
        fconv.close()

        #Validation
        if (epoch+1) % args.test_every == 0:
            print("\nValidation...")
            val_dset.setmode(1)
            probs, val_loss = inference(epoch, val_loader, model, criterion)
            preds, targets = group_max(np.array(val_dset.slideIDX), probs, np.array(val_dset.targets))
            err,fpr,fnr = calc_err(preds, targets)

            # maxs = group_max(np.array(val_dset.slideIDX), probs, len(val_dset.targets))
            # pred = [1 if x >= 0.5 else 0 for x in maxs]
            # err,fpr,fnr = calc_err(pred, val_dset.targets)
            print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, err, fpr, fnr))
            fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
            fconv.write('{},error,{}\n'.format(epoch+1, err))
            fconv.write('{},fpr,{}\n'.format(epoch+1, fpr))
            fconv.write('{},fnr,{}\n'.format(epoch+1, fnr))
            fconv.close()

            #Save best model
            if val_loss <= best_loss:
                best_loss = val_loss
                acc = 1-(fpr+fnr)/2.

                obj = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'accuracy': acc,
                    'optimizer' : optimizer.state_dict()
                }
                torch.save(obj, os.path.join(args.output,'checkpoint_best_loss_%.4f_acc_%.2f.pth' % (best_loss, acc)))
                print("Best loss updated (Epoch %s | Loss %.4f | Accuracy %.2f)" % (epoch+1, best_loss, acc))


def inference(run, loader, model, criterion):
    model.eval()
    running_loss = 0.

    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            if (i+1) % 10==0:
                print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))
                # break
            input = input.cuda().float()
            target = target.cuda()
            output = model(input)

            loss = criterion(output, target)
            running_loss += loss.item()*input.size(0)

            output = F.softmax(output, dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()

    return probs.cpu().numpy(), running_loss/len(loader.dataset)


def train(run, loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.

    cnt = 0
    for i, (input, target) in enumerate(loader):
        input = input.cuda().float()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)

        for pred, true in zip(output, target):
            # print('%d | Pred : %s | GT : %s' % (cnt, F.softmax(pred, dim=0)[1].item(), true.item()))
            cnt += 1
    return running_loss/len(loader.dataset)

def calc_err(pred,real):
    # print("Validation\n>>> Pred : %s\n>>> GT :%s\n"%(pred, real))
    pred = np.array(pred>0.5).astype(int)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    return err, fpr, fnr

def group_argtopk(groups, data,k=1):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])

# def group_max(groups, data, nmax):
#     out = np.empty(nmax)
#     out[:] = np.nan
#     order = np.lexsort((data, groups))
#     groups = groups[order]
#     data = data[order]
#     index = np.empty(len(groups), 'bool')
#     index[-1] = True
#     index[:-1] = groups[1:] != groups[:-1]
#     out[groups[index]] = data[index]
#     return out

def group_max(groups, data, targets):
    nmax = len(targets)
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
    targets = targets.copy()[order][index]

    for i, (pred, true) in enumerate(zip(out, targets)):
        print('%d | Pred : %s | GT : %s' % (i, pred, true))

    # out = (out > 0.5).astype(np.uint8)
    return out, targets

class MILTraindataset(data.Dataset):
    def __init__(self, args, transform=None):
        self.train_dir = args.train_dir

        #Flatten grid
        patch_paths = []
        slideIDX = []
        targets = []

        print("Generating Train Dataset...")
        img_list = sorted(glob(os.path.join(args.train_dir, '*', '*.png')))
        for patch_path in tqdm(img_list, desc='Training Dataset'):
            h_idx, w_idx = map(int, patch_path.rstrip('.png').split('_')[-2:])
            include = args.overlap or (h_idx%4==0 & w_idx%4==0)

            if include:
                slide_idx = self.get_slide_idx_from_path(patch_path)
                slideIDX.append(slide_idx)
                patch_paths.append(patch_path)
                targets.append(int(patch_path.split(os.sep)[-2]))

        print('Number of patches: {}'.format(len(patch_paths)))

        self.slideIDX = slideIDX
        self.targets = targets
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

    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x],self.patch_paths[x],self.targets[x]) for x in idxs]

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def __getitem__(self,index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            patch_path = self.patch_paths[index]

            img = self.read_img(patch_path)
            if self.transform is not None:
                img = self.transform(img)
            return img, self.targets[index]

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


class MILValiddataset(data.Dataset):
    def __init__(self, args, transform=None):
        self.valid_dir = args.valid_dir
        self.label_dict = dict(pd.read_csv(args.valid_annot).values)
        
        #Flatten grid
        patch_paths = []
        slideIDX = []
        targets = []

        print("Generating Valid Dataset...")
        img_list = sorted(glob(os.path.join(args.valid_dir, '*', 'test', '*.png')))
        for patch_path in tqdm(img_list, desc='Validation Dataset'):
            h_idx, w_idx = map(int, patch_path.rstrip('.png').split('_')[-2:])
            include = args.overlap or (h_idx%4==0 & w_idx%4==0)

            if include:
                slide_idx = self.get_slide_idx_from_path(patch_path)
                slideIDX.append(slide_idx)
                patch_paths.append(patch_path)
                target = self.label_dict[patch_path.split(os.sep)[-3]]
                targets.append(int())
            

        print('Number of patches: {}'.format(len(patch_paths)))

        self.slideIDX = slideIDX
        self.targets = targets
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

    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x],self.patch_paths[x],self.targets[x]) for x in idxs]

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def __getitem__(self,index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            patch_path = self.patch_paths[index]

            img = self.read_img(patch_path)
            if self.transform is not None:
                img = self.transform(img)
            return img, self.targets[index]

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
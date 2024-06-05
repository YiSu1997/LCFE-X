import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from HyperchannelViT import hyperchannelViT
from sklearn.metrics import confusion_matrix
import numpy as np
import time
import os
import auxil
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from draw_maps import draw_classification
import pandas as pd
from morphFormer import MorphFormer

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['PU', 'HS'], default='PU', help='dataset to use')
parser.add_argument('--tr_percent', type=float, default=0.1, help='percent of train dataset')
parser.add_argument('--mode', choices=['None', 'LCFE'], default='LCFE', help='enhanced mode choice')
parser.add_argument('--backbone', choices=['ViT', 'SF', 'MF'], default='ViT', help='basic model choice')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=300, help='number of evaluation')
parser.add_argument('--patches', type=int, default=1, help='size of patches')
parser.add_argument('--group', type=int, default=1, help='number of related band')
parser.add_argument('--epoches', type=int, default=300, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


# -------------------------------------------------------------------------------
# get local 3D patches
def gain_neighborhood_band(x_train, band, group, patch=5):
    nn = group // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*group, band),dtype=float)
    # central area
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    # left mirror area
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    # right mirror area
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]

    return x_train_band  # [samples, group*group, bands]


def gain_neighborhood_overlappingbandgroup(x_train, band, group, patch=5):
    # overlapping band grouping
    nn = group // 2
    pp = (patch*patch) // 2
    x_train_band = np.zeros((x_train.shape[0], patch, patch, group, band),dtype=float)
    x_train = x_train.unsqueeze(3)
    # central area
    x_train_band[:,:,:,nn:(nn+1),:] = x_train
    # left mirror area
    for i in range(nn):
        if pp > 0:
            x_train_band[:,:,:,i:(i+1),:i+1] = x_train[:,:,:,:,band-i-1:]
            x_train_band[:,:,:,i:(i+1),i+1:] = x_train[:,:,:,:,:band-i-1]
        else:  # patch=1
            x_train_band[:,:,:,i:(i+1),:(nn-i)] = x_train[:,0:1,0:1,:,(band-nn+i):]
            x_train_band[:,:,:,i:(i+1),(nn-i):] = x_train[:,0:1,0:1,:,:(band-nn+i)]
    # right mirror area
    for i in range(nn):
        if pp > 0:
            x_train_band[:,:,:,(nn+i+1):(nn+i+2),:band-i-1] = x_train[:,:,:,:,i+1:]
            x_train_band[:,:,:,(nn+i+1):(nn+i+2),band-i-1:] = x_train[:,:,:,:,:i+1]
        else:
            x_train_band[:,:,:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train[:,0:1,0:1,:,:(i+1)]
            x_train_band[:,:,:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train[:,0:1,0:1,:,(i+1):]

    return x_train_band

# -------------------------------------------------------------------------------
# create HSI Cubes with all bands
def load_hyper(args):
    pixelsO, spaceO, labelsO, numclass = auxil.loadData(args.dataset)
    pixelsO, spaceO, labelsO = auxil.createImageCubes(pixelsO, spaceO, labelsO, windowSize=args.patches, removeZeroLabels=False)  # 读取3D图像块
    pixels = pixelsO[labelsO != 0]
    space = spaceO[labelsO != 0]
    labels = labelsO[labelsO != 0] - 1
    del pixelsO
    del spaceO
    del labelsO
    bands = pixels.shape[-1];
    numberofclass = len(np.unique(labels))
    pixels = pixels.reshape((pixels.shape[0], args.patches, args.patches, bands, 1))

    return pixels, space, labels, numberofclass, bands


# -------------------------------------------------------------------------------
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# -------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


# -------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    pre_pro = []  # output class probability
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        # compute output
        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
        batch_pred = torch.softmax(batch_pred.cpu(), dim=1).cuda(0)
        pre_pro.append(batch_pred.cpu().detach().numpy())

    return top1.avg, objs.avg, tar, pre, pre_pro, train_loss


# -------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, criterion):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    pre_pro = []  # output class probability
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data.data)

        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
        batch_pred = torch.softmax(batch_pred.cpu(), dim=1).cuda(0)
        pre_pro.append(batch_pred.cpu().detach().numpy())

    return tar, pre, pre_pro


# -------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


# -------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


# -------------------------------------------------------------------------------
# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

# -------------------------------------------------------------------------------
# print and save param settings
model_file = "./SavedModel/{}/{}_{}/".format(args.dataset, args.mode, args.backbone)
if not os.path.exists(model_file):
    os.makedirs(model_file)
auxil.print_args(vars(args), model_file)
# -------------------------------------------------------------------------------
# obtain train and test data
pixels, space, labels, num_classes, band = load_hyper(args)
# ten-fold cross-validation
split = StratifiedShuffleSplit(n_splits=10, test_size=1 - args.tr_percent, random_state=args.seed)
split_id = 0
for train_index, test_index in split.split(pixels, labels):
    split_id += 1
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>split %d<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<' % split_id)
    x_train = pixels[train_index]
    y_train = labels[train_index]
    x_test = pixels[test_index]
    testSpace = space[test_index]
    y_test = labels[test_index]

    n, m = x_train.shape[0], x_test.shape[0]  # Number of training samples

    # shuffle training set by random
    shuffleIndex = np.random.choice(np.arange(n), size=(n,), replace=False)
    x_train = x_train[shuffleIndex]
    y_train = y_train[shuffleIndex]

# -------------------------------------------------------------------------------
    # reshape
    x_train = torch.squeeze(torch.from_numpy(x_train), 4)
    x_test = torch.squeeze(torch.from_numpy(x_test), 4)
    if args.mode == 'None':
        # No channel feature extraction, flatten directly
        x_train = torch.from_numpy(gain_neighborhood_band(x_train, band, args.group, patch=args.patches).transpose(0, 2, 1)).type(torch.FloatTensor)  # [samples, group*group, bands]
        x_test = torch.from_numpy(gain_neighborhood_band(x_test, band, args.group, patch=args.patches).transpose(0, 2, 1)).type(torch.FloatTensor)
    else:
        # overlapping grouping, and use LCFE to extracte local channel features
        x_train = torch.tensor(gain_neighborhood_overlappingbandgroup(x_train, band, args.group, patch=args.patches)).permute(0, 4, 1, 2, 3).type(torch.FloatTensor)  # [batch_num,group_num,patch_size,patch_size,group]
        x_test = torch.tensor(gain_neighborhood_overlappingbandgroup(x_test, band, args.group, patch=args.patches)).permute(0, 4, 1, 2, 3).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    Label_train = Data.TensorDataset(x_train, y_train)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)
    Label_test = Data.TensorDataset(x_test, y_test)

    label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
    label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=False)

# -------------------------------------------------------------------------------
    # create model
    if args.backbone == 'SF':
        backbone = 'CAF'
    else:
        backbone = args.backbone
    if args.backbone == 'ViT' or args.backbone == 'SF':
        model = hyperchannelViT(
            patch_size=args.patches,
            near_band=args.group,  # the number of grouped bands
            band=band,  # the number of bands in hyperspectral image
            num_classes=num_classes,
            dim=64,
            depth=5,
            heads=4,
            mlp_dim=8,
            dropout=0.1,
            emb_dropout=0.1,
            mode=args.mode,  # None/LCFE
            backbone=backbone  # ViT/CAF(SF)
        )
    elif backbone == 'MF':
        model = MorphFormer(
            FM=16,
            NC=band,
            num_classes=num_classes,
            patch_size=args.patches,
            near_band=args.group,
            mode=args.mode
        )

    model = model.cuda()
    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)
    # -------------------------------------------------------------------------------
    print("start training")
    tic = time.time()
    train_loss = []
    iteration = 1  # batch numbers

    for epoch in range(args.epoches):
        # train model
        model.train()
        train_acc, train_obj, tar_t, pre_t, pre_pro_t, train_loss = train_epoch(model, label_train_loader, criterion, optimizer)
        scheduler.step()
        train_report_dict = classification_report(tar_t, pre_t, output_dict=True, digits=4, zero_division=1)
        print(
            "Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
                .format(epoch + 1, train_obj, train_acc))

        # output test results based on the set test frequency
        if epoch != 0 and ((epoch % args.test_freq == 0) | (epoch == args.epoches - 1)):
            model.eval()
            tar_v, pre_v, pre_pro_v = valid_epoch(model, label_test_loader, criterion)
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)


    toc = time.time()
    runtime = toc - tic
    print("Running Time: {:.2f}".format(runtime))
    print("**************************************************")


    print("Final result:")
    auxil.print_report(tar_v, pre_v, num_classes)
    draw_classification(args, pre_v, testSpace, tar_v, split_id, model_file, 'svg')  # the final variable is the storage format
    # save result
    reportFile = "{}/{}_{}-{}_patches={}_band={}.csv".format(model_file, args.dataset, args.mode, args.backbone, args.patches, args.group)
    report_df = pd.DataFrame(classification_report(tar_v, pre_v, output_dict=True, digits=4, zero_division=1)).T
    report_df = auxil.save_classification_report(report_df, Kappa2, runtime, reportFile)
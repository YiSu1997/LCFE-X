import os
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import pandas as pd


def loadData(name):
    data_path = os.path.join(os.getcwd(),'data/')
    if name in ['PU']:
        data = sio.loadmat(os.path.join(data_path, 'paviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']
    elif name in ['HS']:
        data = sio.loadmat(os.path.join(data_path, 'Houston.mat'))['input']
        labels = sio.loadmat(os.path.join(data_path, 'Houston_gt.mat'))['gt']
    else:
        print("NO DATASET")
        exit()

    shapeor = data.shape
    data = data.reshape(-1, data.shape[-1])
    data = MinMaxScaler(feature_range=(-1, 1)).fit_transform(data)  # normalization
    data = data.reshape(shapeor)
    # calculate coordinate matrix
    space=np.zeros((data.shape[0],data.shape[1], 2))  # the third dimension consists of two columns of coordinates
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
                space[i,j,0] = i  # X-axis
                space[i,j,1] = j  # Y-axis
    num_class = len(np.unique(labels)) - 1  # count the total class number

    return data, space, labels, num_class


def padWithZeros(X, margin=2):
    # pad with zeros
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


# boundary expansion: mirror
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    # central area
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    # left mirror area
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    # right mirror area
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    # upper mirror area
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    # lower mirror area
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi


def createImageCubes(X, s, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = mirror_hsi(X.shape[0], X.shape[1], X.shape[2], X, patch=windowSize)  # boundary expansion: mirror
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2])).astype("float32")
    patchesSpace = np.zeros((X.shape[0] * X.shape[1], 2))  # coordinates
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch.astype("float32")
            patchesSpace[patchIndex, :] = s[r-margin, c-margin]  # get the coordinates of the sample
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        # remove background points
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesSpace = patchesSpace[patchesLabels>0,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesSpace.astype("int"), patchesLabels.astype("int")


def print_args(args,model_file):
    with open(model_file + 'param_set.txt', 'w') as param:
        for k, v in zip(args.keys(), args.values()):
            print("{0}: {1}".format(k, v))
            param.write('--{0}={1} '.format(k, v))


def print_report(true, pred_tar, num_classes):
    # input class probability and print classification report
    target_names = ['{}'.format(i) for i in range(num_classes)]
    report = classification_report(true, pred_tar, target_names=target_names, digits=4)
    print(report)

    return report


def save_classification_report(report_df1, Kappa, time_c, save_path):
    # save classification report to .csv file
    report_df1.iloc[-3,:2]= np.nan
    report_df1.iloc[-3,3]= report_df1.iloc[-2,3]
    report_df2 = pd.Series([Kappa, time_c], index=['kappa','time'])
    report_df = pd.concat([report_df1, report_df2], ignore_index=False)

    report_df.to_csv(save_path, mode='a')

    return report_df.round(4)



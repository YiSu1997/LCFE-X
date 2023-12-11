import matplotlib.pyplot as plt
import numpy as np
import spectral
import os
from scipy.io import loadmat
import cv2
import matplotlib.patches as mpatches

# draw in different color lists
pu_color = np.array([[255, 255, 255],
                     [184, 40, 99],
                     [74, 77, 145],
                     [35, 102, 193],
                     [238, 110, 105],
                     [117, 249, 76],
                     [114, 251, 253],
                     [126, 196, 59],
                     [234, 65, 247],
                     [141, 79, 77],
                     [183, 40, 99],
                     [0, 39, 245],
                     [90, 196, 111],
                     [50, 140, 100],
                     [70, 140, 200],
                     [100, 150, 170]])

hs_color = np.array([[255, 255, 255],  # 白色背景
                     [144, 238, 144],  # Healthy grass淡绿色
                     [50, 205, 50],  # Stressed grass酸橙绿
                     [173, 255, 47],  # Synthetic grass绿黄色
                     [0, 100, 0],  # Trees深绿色
                     [189, 183, 107],  # Soil深卡其色
                     [0, 191, 255],  # Water深天蓝
                     [128, 138, 135],  # Residential冷灰
                     [0, 51, 102],  # Commercial 深灰蓝
                     [188, 143, 143],  # Road 玫瑰棕色
                     [178, 34, 34],  # Highway 耐火砖
                     [255, 69, 0],  # Railway 橙红色
                     [255, 215, 0],  # Parking Lot 1 12 金色
                     [244, 164, 96],  # Parking Lot 2 13 沙棕色
                     [32, 178, 170],  # Tennis Court 浅海洋绿
                     [220, 20, 60]])  # Running Track 猩红


def draw_classification(args, final_prediction, s_test, y_test, iter, mapPath, savefomat):
    # recommend set savefomat as 'svg' or 'jpg'
    classification_result = loadgt(args.dataset)  # initialize the ground truth matrix
    if args.dataset == 'HS':
        color_type = hs_color
    elif args.dataset == 'PU':
        color_type = pu_color
    else:
        print("NO DATASET COLOR")
        exit()
    # get classification result matrix
    for i in range(len(y_test)):
        # test and update the label in matrix for misclassification
        if int(y_test[i]) != final_prediction[i]:
            # get corresponding position
            pos_x = int(s_test[i][0])
            pos_y = int(s_test[i][1])
            classification_result[pos_x][pos_y] = final_prediction[i]
        else:
            continue

    ground_classification = spectral.imshow(classes = classification_result.astype(int),figsize =(9,9),colors=color_type)
    # save map
    plt.show()
    if not os.path.exists(mapPath):
        os.makedirs(mapPath)
    plt.savefig(mapPath+'iter{}.{}'.format(iter, savefomat),dpi=1200, bbox_inches = 'tight')

def loadgt(file_name):
    file_dir = os.getcwd()
    file_path = os.path.join(file_dir, 'data')
    if file_name == 'PU':
        gt = loadmat(os.path.join(file_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif file_name == 'HS':
        gt = loadmat(os.path.join(file_path, 'Houston_gt.mat'))['gt']
    else:
        print("NO DATASET GROUNDTRUTH")
        exit()

    return gt


def draw_goundtruth(file_name):
    output_image = loadgt(file_name)
    print(output_image)
    if file_name == 'PU':
        color_type = pu_color
    elif file_name == 'HS':
        color_type = hs_color
    else:
        print("NO DATASET COLOR")
        exit()
    ground_truth = spectral.imshow(classes=output_image.astype(int),figsize =(9,9), colors=color_type)
    plt.show()
    plt.savefig('./SavedModel/{}/gt.svg'.format(file_name), dpi=1200,bbox_inches='tight')

    cv2.imshow('1', output_image)
    cv2.waitKey(0)

def draw_labels(file_name):
    pu_color = [
                [184, 40, 99, 255],
                [74, 77, 145, 255],
                [35, 102, 193, 255],
                [238, 110, 105, 255],
                [117, 249, 76, 255],
                [114, 251, 253, 255],
                [126, 196, 59, 255],
                [234, 65, 247, 255],
                [141, 79, 77, 255],
                [183, 40, 99, 255],
                [0, 39, 245, 255],
                [90, 196, 111, 255],
                [50, 140, 100, 255],
                [70, 140, 200, 255],
                [100, 150, 170, 255]]

    hs_color = np.array([[144, 238, 144, 255],  # Healthy grass淡绿色
                         [50, 205, 50, 255],  # Stressed grass酸橙绿
                         [173, 255, 47, 255],  # Synthetic grass绿黄色
                         [0, 100, 0, 255],  # Trees深绿色
                         [189, 183, 107, 255],  # Soil深卡其色
                         [0, 191, 255, 255],  # Water深天蓝
                         [128, 138, 135, 255],  # Residential冷灰
                         [0, 51, 102, 255],  # Commercial 深灰蓝
                         [188, 143, 143, 255],  # Road 玫瑰棕色
                         [178, 34, 34, 255],  # Highway 耐火砖
                         [255, 69, 0, 255],  # Railway 橙红色
                         [255, 215, 0, 255],  # Parking Lot 1 12 金色
                         [244, 164, 96, 255],  # Parking Lot 2 13 沙棕色
                         [32, 178, 170, 255],  # Tennis Court 浅海洋绿
                         [220, 20, 60, 255]])  # Running Track 猩红

    pu_label = ['Asphalt','Meadows','Gravel','Trees','Painted metal sheets','Bare Soil','Bitumen','Self-Blocking Bricks','Shadows']
    hs_label = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees', 'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway',
                'Railway', 'Parking Lot 1', 'Parking Lot 2', 'Tennis Court', 'Running Track']

    # initialize the classification result matrix and set colors
    if file_name == 'PU':
        color_type = pu_color
        label_type = pu_label
    elif file_name == 'HS':
        color_type = hs_color
        label_type = hs_label
    else:
        print("NO DATASET COLOR")
        exit()

    patches = [mpatches.Patch(edgecolor='black',color=[x/255 for x in color_type[i]],label="{:s}".format(label_type[i]) ) for i in range(len(label_type))]

    ax = plt.gca()
    # bbox_to_anchor specifies the position of the legend
    if file_name == 'PU':
        ax.legend(handles=patches, bbox_to_anchor=(0.5, 0.5), ncol=1)
    elif file_name == 'HS':
        ax.legend(handles=patches, bbox_to_anchor=(1, 1), ncol=3)
    ax.axis("off")  # remove coordinate
    plt.savefig('./SavedModel/{}/labels.svg'.format(file_name), dpi=1200, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    draw_goundtruth('HS')
    draw_labels('HS')

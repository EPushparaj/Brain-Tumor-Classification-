import os
from Model_ADSL_New import Model_ADSL
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2 as cv
import numpy as np
from datashader import Point
from numpy import matlib
import random as rn
from EVO import EVO
from Global_Vars import Global_Vars
from Image_Results import Image_Results
from Model_ALEXNET import Model_ALEXNET
from Model_GoogleNet import Model_GoogleNet
from Model_HDCNet import Model_HDCNet
from Model_Trans_Unet import Model_Trans_Unet
from Morphological_features import Extract_Morphological_features
from Objective_Function import Objfun_Cls, Objective_Seg, Objfun_Feat
from PROPOSED import PROPOSED
from RSA import RSA
from Region_growing import Region_Growing
from SCO import SCO
from TFMO import TFMOA
from Texture_Feature import Extract_Texture_features
from plot_Segment import Plot_seg_Results
from Plot_Results import plot_results_conv, Confusion_matrix, Plot_ROC, plot_results

# Read Dataset
an = 0
if an == 1:
    Image_Output = []
    GT_output = []
    Target = []
    path = './Datasets/Dataset_1'
    data_dir = os.listdir(path)
    for data in range(len(data_dir)):
        out_dir = path + '/' + data_dir[data]
        folder = os.listdir(out_dir)
        for i in range(len(folder)):
            print(data, i)
            in_dir = out_dir + '/' + folder[i]
            img = cv.imread(in_dir)
            re_Img = cv.resize(img, (256, 256))
            if in_dir.endswith('_mask.png'):
                GT_output.append(re_Img)
            else:
                Image_Output.append(re_Img)
                tar = data_dir[data]
                Target.append(tar)
    Targ = np.asarray(Target)
    uni = np.unique(Targ)
    tar = np.zeros((Targ.shape[0], len(uni))).astype('int')
    for i in range(len(uni)):
        ind = np.where((Targ == uni[i]))
        tar[ind[0], i] = 1
    np.save('Images.npy', Image_Output)
    np.save('Ground_Truth.npy', GT_output)
    np.save('Target.npy', tar)

# Optimization for Segmentation
an = 0
if an == 1:
    sol = []
    fitness = []
    Images = np.load('Images.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_Vars.Images = Images
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 2
    xmin = matlib.repmat(([0]), Npop, 1)
    xmax = matlib.repmat(([255]), Npop, 1)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])
    fname = Objective_Seg
    max_iter = 50

    print('RSA....')
    [bestfit1, fitness1, bestsol1, Time1] = RSA(initsol, fname, xmin, xmax, max_iter)

    print('TFMOA....')
    [bestfit2, fitness2, bestsol2, Time2] = TFMOA(initsol, fname, xmin, xmax, max_iter)

    print('SCO....')
    [bestfit3, fitness3, bestsol3, Time3] = SCO(initsol, fname, xmin, xmax, max_iter)

    print('EVO....')
    [bestfit4, fitness4, bestsol4, Time4] = EVO(initsol, fname, xmin, xmax, max_iter)

    print('PROPOSED....')
    [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

    sol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]

    np.save('Bestsol_1.npy', sol)

# Segmentation for TUNet-MRG
an = 0
if an == 1:
    Segmented = []
    img = np.load('Images.npy', allow_pickle=True)
    GT = np.load('Ground_Truth.npy', allow_pickle=True)
    sol = np.load('Bestsol_1.npy', allow_pickle=True)[4, :]
    Images = Model_Trans_Unet(img, GT, sol)
    for n in range(len(Images)):
        print(n + 1)
        image = Images[n]
        ## Thesholding ##
        ret, thresh_image = cv.threshold(image, sol[0].astype('int'), 255, cv.THRESH_BINARY)
        Thresh_Img = np.zeros((thresh_image.shape)).astype(np.uint8)
        pn = np.where(thresh_image == 1)
        Thresh_Img[pn] = 255
        ##### Region Growing ####
        seeds = [Point(10, 10), Point(82, 150), Point(sol[1].astype('int'), sol[2].astype('int'))]  # [Point(0, 255)]
        binaryImg = Region_Growing(Thresh_Img, seeds)  # Region Growing
        region = np.zeros((binaryImg.shape)).astype(np.uint8)
        pn = np.where(binaryImg == 1)
        region[pn] = 255
        Fusion_image = ((Thresh_Img + region) / 2).astype('uint8')
        Segmented.append(Fusion_image)
    np.save('Trans_UNet_RG.npy', Segmented)

# Optimization for Feature Selection:
an = 0
if an == 1:
    Feature = np.load('Trans_UNet_RG.npy', allow_pickle=True)
    Feature_reshape = np.reshape(Feature, (Feature.shape[0], Feature.shape[1] * Feature.shape[2] * Feature.shape[3]))
    Target = np.load('Target.npy', allow_pickle=True)
    Global_Vars.Images = Feature
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 10
    xmin = matlib.repmat((np.zeros(10)), Npop, 1)
    xmax = matlib.repmat((Feature_reshape.shape[1] - 1 * np.ones(10)), Npop, 1)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])
    fname = Objfun_Feat
    max_iter = 50

    print('RSA....')
    [bestfit1, fitness1, bestsol1, Time1] = RSA(initsol, fname, xmin, xmax, max_iter)

    print('TFMOA....')
    [bestfit2, fitness2, bestsol2, Time2] = TFMOA(initsol, fname, xmin, xmax, max_iter)

    print('SCO....')
    [bestfit3, fitness3, bestsol3, Time3] = SCO(initsol, fname, xmin, xmax, max_iter)

    print('EVO....')
    [bestfit4, fitness4, bestsol4, Time4] = EVO(initsol, fname, xmin, xmax, max_iter)

    print('PROPOSED....')
    [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

    sol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]

    np.save('Bestsol_2.npy', sol)

### Feature Selection
an = 0
if an == 1:
    Feat = np.load('Trans_UNet_RG.npy', allow_pickle=True)
    Feature = np.reshape(Feat, (Feat.shape[0], Feat.shape[1] * Feat.shape[2] * Feat.shape[3]))
    sol = np.load('Bestsol_2.npy', allow_pickle=True)[4, :]
    Selected_Feature = Feature[:, np.round(sol).astype('int')]
    np.save('Features_Selected.npy', Selected_Feature)

# Feature Extraction
an = 0
if an == 1:
    Feat = np.load('Trans_UNet_RG.npy', allow_pickle=True)
    Feature_1 = Extract_Texture_features(Feat)
    Feature_2 = Extract_Morphological_features(Feat)
    np.save('Texture_Feature.npy', Feature_1)
    np.save('Morphological_Feature.npy', Feature_2)

# Feature Concatenation
an = 0
if an == 1:
    Feat_1 = np.load('Texture_Feature.npy', allow_pickle=True)
    Feat_1_re = np.reshape(Feat_1, (Feat_1.shape[0], Feat_1.shape[1] * Feat_1.shape[2]))
    Feat_2 = np.load('Morphological_Feature.npy', allow_pickle=True)
    Feat_3 = np.load('Features_Selected.npy', allow_pickle=True)
    Feat = np.concatenate((Feat_1_re, Feat_2, Feat_3), axis=1)
    np.save('Feature.npy', Feat)

# Optimization for Classification
an = 0
if an == 1:
    Images = np.load('Feature.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_Vars.Images = Images
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3
    xmin = matlib.repmat(([5, 5, 0]), Npop, 1)
    xmax = matlib.repmat(([255, 50, 4]), Npop, 1)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])
    fname = Objfun_Cls
    max_iter = 50

    print('RSA....')
    [bestfit1, fitness1, bestsol1, Time1] = RSA(initsol, fname, xmin, xmax, max_iter)

    print('TFMOA....')
    [bestfit2, fitness2, bestsol2, Time2] = TFMOA(initsol, fname, xmin, xmax, max_iter)

    print('SCO....')
    [bestfit3, fitness3, bestsol3, Time3] = SCO(initsol, fname, xmin, xmax, max_iter)

    print('EVO....')
    [bestfit4, fitness4, bestsol4, Time4] = EVO(initsol, fname, xmin, xmax, max_iter)

    print('PROPOSED....')
    [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

    sol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    fitness = [fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()]

    np.save('Bestsol_3.npy', sol)
    np.save('Fitness.npy', fitness)

## Classification ##
an = 0
if an == 1:
    Eval = []
    Images = np.load('Feature.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    sol = np.load('Bestsol_3.npy', allow_pickle=True)
    Batch_Size = ['4', '8', '16', '32', '64']
    for m in range(len(Batch_Size)):
        per = round(Images.shape[0] * 0.75)
        EVAL = np.zeros((10, 25))
        for i in range(5):  # for all algorithms
            train_data = Images[:per, :]
            train_target = Target[:per, :]
            test_data = Images[per:, :]
            test_target = Target[per:, :]
            EVAL[i, :] = Model_ADSL(train_data, train_target, test_data, test_target, sol[i].astype('int'))
        train_data = Images[:per, :]
        train_target = Target[:per, :]
        test_data = Images[per:, :]
        test_target = Target[per:, :]
        EVAL[5, :] = Model_GoogleNet(train_data, train_target, test_data, test_target)
        EVAL[6, :] = Model_ALEXNET(train_data, train_target, test_data, test_target)
        EVAL[7, :] = Model_HDCNet(train_data, train_target, test_data, test_target)
        EVAL[8, :] = Model_ADSL(train_data, train_target, test_data, test_target)
        EVAL[9, :] = EVAL[4, :]
        Eval.append(EVAL)
    np.save('Eval_all_Batchsize.npy', Eval)

Plot_seg_Results()
plot_results_conv()
Confusion_matrix()
Plot_ROC()
plot_results()
Image_Results()

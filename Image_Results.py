import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def Image_Results():
    Orig = np.load('Ori.npy', allow_pickle=True)
    G_T =  np.load('GT.npy', allow_pickle=True)
    Image1 = np.load('3DUNet.npy', allow_pickle=True)
    Image2 = np.load('UNet.npy', allow_pickle=True)
    Image3 = np.load('TransUNet.npy', allow_pickle=True)
    Image4 = np.load('UNet_RG.npy', allow_pickle=True)
    segment = np.load('PROPOSED.npy', allow_pickle=True)
    for j in range(5):
        original = Orig[j]
        GT = G_T[j]
        image1 = Image1[j]
        image2 = Image2[j]
        image3 = Image3[j]
        image4 = Image4[j]
        seg = segment[j]
        Output1 = np.zeros((image1.shape)).astype('uint8')
        ind1 = np.where(image1 > 0)
        Output1[ind1] = 255

        Output2 = np.zeros((image2.shape)).astype('uint8')
        ind2 = np.where(image2 > 0)
        Output2[ind2] = 255

        Output3 = np.zeros((image3.shape)).astype('uint8')
        ind3 = np.where(image3 > 0)
        Output3[ind3] = 255

        Output4 = np.zeros((image4.shape)).astype('uint8')
        ind4 = np.where(image4 > 0)
        Output4[ind4] = 255

        Output5 = np.zeros((seg.shape)).astype('uint8')
        ind5 = np.where(seg > 0)
        Output5[ind5] = 255

        Output6 = np.zeros((GT.shape)).astype('uint8')
        ind6 = np.where(GT > 0)
        Output6[ind6] = 255

        fig, ax = plt.subplots(1, 4)
        plt.suptitle("Image %d" % (j + 1), fontsize=20)
        plt.subplot(2, 3, 1)
        plt.title('Original')
        plt.imshow(original)
        plt.subplot(2, 3, 2)
        plt.title('3DUNet')
        plt.imshow(Output1)
        plt.subplot(2, 3, 3)
        plt.title('UNet')
        plt.imshow(Output2)

        plt.subplot(2, 3, 4)
        plt.title('TransUNet')
        plt.imshow(Output3)

        plt.subplot(2, 3, 5)
        plt.title('UNet-RG')
        plt.imshow(Output4)

        plt.subplot(2, 3, 6)
        plt.title('PROPOSED')
        plt.imshow(Output5)
        plt.show()


        cv.imwrite('./Seg/NewSeg/Original-' + str(j + 1) + '.png', original)
        cv.imwrite('./Seg/NewSeg/GT-' + str(j+1) + '.png', Output6)
        cv.imwrite('./Seg/NewSeg/3DUNet-' + str(j + 1) + '.png', Output1)
        cv.imwrite('./Seg/NewSeg/UNet-' + str(j + 1) + '.png', Output2)
        cv.imwrite('./Seg/NewSeg/TransUNet-' + str(j + 1) + '.png', Output3)
        cv.imwrite('./Seg/NewSeg/UNet-RG-' + str(j + 1) + '.png', Output4)
        cv.imwrite('./Seg/NewSeg/Trans-UNet-RG-' + str(j + 1) + '.png', Output5)



if __name__ == '__main__':
    Image_Results()



import numpy as np
import cv2 as cv


def Extract_Morphological_features(images):
    morphological1 = []
    for j in range(len(images)):
        # $$ Morphological Feature $$#
        # $$ Erosion $$#
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv.erode(images[j], kernel, iterations=1)
        # $$ Dilation $$#
        dilation = cv.dilate(images[j], kernel, iterations=1)
        # $$ Opening ##$
        opening = cv.morphologyEx(images[j], cv.MORPH_OPEN, kernel)
        # $$ Closing $$#
        closing = cv.morphologyEx(images[j], cv.MORPH_CLOSE, kernel)
        # $$ Morphological Gradient $$#
        gradient = cv.morphologyEx(images[j], cv.MORPH_GRADIENT, kernel)
        # $$ Top Hat $$3
        tophat = cv.morphologyEx(images[j], cv.MORPH_TOPHAT, kernel)
        # $$ Black Hat $$#
        blackhat = cv.morphologyEx(images[j], cv.MORPH_BLACKHAT, kernel)
        morphological1.append(np.asarray(
            [erosion.ravel(), dilation.ravel(), opening.ravel(), closing.ravel(), gradient.ravel(), tophat.ravel(),
             blackhat.ravel()]).ravel())
    return morphological1
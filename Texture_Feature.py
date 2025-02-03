
from GLCM import Image_GLCM
from LWP import get_lwp_descriptor
import cv2 as cv

def Extract_Texture_features(images):
    Glcm = []
    LWP = []
    for j in range(len(images)):
        print(j)
        # $$ Texture Feature $$#
        img = cv.cvtColor(images[j], cv.COLOR_BGR2GRAY)
        GLCM = Image_GLCM(img)
        Lwp = get_lwp_descriptor(GLCM)
        LWP.append(Lwp.astype('int'))
        # Glcm.append(GLCM.astype('int'))
    # feat = np.concatenate((Glcm, LWP), axis=0)
    return LWP
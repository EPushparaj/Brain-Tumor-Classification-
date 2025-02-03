import numpy as np
from skimage.feature import graycomatrix, graycoprops


def Image_GLCM(image):
    # Define the properties to calculate
    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # Compute the GLCM
    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)

    # Calculate the contrast, dissimilarity, homogeneity, energy, correlation
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')

    return glcm[:, :, 0, 0]

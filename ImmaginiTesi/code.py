import numpy as np
import matplotlib.pyplot as plt

mri_filepath = "/Users/francesco/Tesi/BC-ML4/ImmaginiTesi/ambl_1.npy"
roi_filepath = "/Users/francesco/Tesi/BC-ML4/ImmaginiTesi/ROI 0.npy"

mri = np.load(mri_filepath)
roi = np.load(roi_filepath)

# pick the non zero indexes on the roi mask
z_indexes, y_indexes, x_indexes = roi.nonzero()
z_indexes = np.unique(z_indexes)
y_indexes = np.unique(y_indexes)
x_indexes = np.unique(x_indexes)

z = z_indexes[2] # pick one slice of the roi
plt.figure(figsize = (10,10), dpi=150)
plt.imshow(mri[z, 60:260, 60:260], cmap=plt.cm.gray)
plt.imshow(roi[z,60:260,60:260], cmap=plt.cm.gray, alpha=(roi[z,60:260,60:260] > 0).astype(float) )
plt.show()

from scipy.ndimage import binary_erosion

plt.figure(figsize = (10,10), dpi=150)
plt.imshow(mri[z, 60:260, 60:260], cmap=plt.cm.gray)

mask = roi[z,60:260,60:260]
eroded = binary_erosion(mask)
outline = mask - eroded

plt.imshow(outline, cmap=plt.cm.plasma, alpha=(outline > 0).astype(float) )
plt.show()

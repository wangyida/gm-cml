import os
import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('GTK')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
from sklearn.decomposition import FastICA

mat = np.load("img_recon_sm_CGM/feat_shapenet_sm_00000100.npy")
savename = 'lowdim_CGM_shapenet.png'
data = np.matrix(mat[:, 1:])
labels = np.matrix(mat[:, 0])
avg = np.average(data, 0)
means = data - avg

# Compute PCA
tmp = np.transpose(means) * means / labels.shape[0]
D,V = np.linalg.eig(tmp)

E = V[:,:]
y_pca = np.dot(np.matrix(E), np.transpose(means))
#

# Compute ICA
ica = FastICA(n_components=2, max_iter=1000)
y_ica = ica.fit_transform(np.transpose(means))
y_ica = np.transpose(np.dot(means, y_ica))


# Have colormaps separated into categories:
# http://matplotlib.org/examples/color/colormaps_reference.html
plt.set_cmap('gist_rainbow')
fig = plt.figure(figsize=(10,5))

# Plot for PCA projection
plt.subplot(121, projection='rectilinear', frame_on=False)
plt.scatter(np.squeeze(np.array(y_pca[0, :])),
            np.squeeze(np.array(y_pca[1, :])),
            c=np.squeeze(np.array(labels)),
            s=5, alpha=0.7, marker="o")
plt.xlabel('PCA Projection', fontsize=11)

# Plot for ICA projection
plt.subplot(122, projection='rectilinear', frame_on=False)
plt.scatter(np.squeeze(np.array(y_ica[0, :])),
            np.squeeze(np.array(y_ica[1, :])),
            c=np.squeeze(np.array(labels)),
            s=5, alpha=0.7, marker="o")
plt.xlabel('ICA Projection', fontsize=11)
plt.colorbar()

# plt.show()
plt.savefig(savename, bbox_inches='tight', dpi=150, frameon=False)

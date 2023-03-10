''' 
Example 1: How to construct a CT problem with ASTRA and TIGRE

Author: Maria Knudsen (February 2023)
'''
# %%
# Load packages
import sys
import os

# Set path to scripts
scr_path = os.path.realpath(os.path.dirname(__file__))
sys.path.insert(0, scr_path + '/../src')

# Import the ramaining packages
from ct_setup import *
import csv
import numpy as np
import matplotlib.pyplot as plt


# Load data
reader = csv.reader(open('X_128_shepplogan_rnl03.csv'), delimiter=",")
X = np.array(list(reader)).astype("float32")

plt.figure()
plt.imshow(X)
plt.title('Exact Image of Interior',fontname='cmr10',fontsize=16)
plt.colorbar()
plt.savefig("Ex1_phantom.pdf", format="pdf", bbox_inches="tight")

# CT setup
num_pixels      = np.shape(X)[0]    # Number of pixels in x/y direction (X needs to be square)
num_dets        = num_pixels        # Number of detector pixels
num_angles      = 181               # Number of projection angles
ang_start       = 0                 # Start angle in degress
ang_end         = 360               # End angle in degress
angles          = np.linspace(ang_start,ang_end,num_angles,dtype=int) / 180 * np.pi 

# %% ************************************ ASTRA ************************************
# Parameters for ASTRA
proj_model      = 'linear'          # The projection model: 'line', 'strip', or 'linear'
proj_geom       = 'fanflat'         # The projection geometry: 'parallel' or 'fanflat'
source_origin   = 1000              # Distance from source to origin/center
origin_det      = 0                 # Distance from origin/center to detector
det_width       = 1                 # Detector width
gpu             = True              # Construct unmatched normal equations: 'False' or 'True' 

CT_ASTRA = ct_astra(num_pixels,num_angles,num_dets,angles,proj_model,proj_geom,source_origin,origin_det,det_width,gpu)

# Create Sinogram
_, Bexact = astra.create_sino(X, CT_ASTRA.proj_id)

plt.figure()
plt.imshow(Bexact)
plt.title('Noise Free Sinogram from ASTRA',fontname='cmr10',fontsize=16)
plt.colorbar()
plt.savefig("Ex1_Sinogram_ASTRA.pdf", format="pdf", bbox_inches="tight")

# %% ************************************ TIGRE ************************************
# Parameters for TIGRE
fp_model        = 'Siddon'          # The forward projection model: 'Siddon' or 'interpolated'
bp_model        = 'FDK'             # The back projection model: 'matched' or 'FDK'
proj_geom       = 'cone'            # The projection geometry: 'parallel' or 'cone'
source_origin   = 1000              # Distance from source to origin/center
source_det      = 1000              # Distance from source to detector
det_width       = 1                 # Detector width

# Using the same starting angle as in ASTRA requires the angles to be shifted pi/2:
angles = np.linspace(ang_start,ang_end,num_angles,dtype=int) / 180 * np.pi + np.pi/2

CT_TIGRE = ct_tigre(num_pixels,num_angles,num_dets,angles,fp_model,bp_model,proj_geom,source_origin,source_det,det_width)

bexact = tigre.Ax(X.reshape(CT_TIGRE.geo.nVoxel),CT_TIGRE.geo,angles).reshape(-1)

plt.figure()
plt.imshow(bexact.reshape(num_angles,num_dets))
plt.title('Noise Free Sinogram from TIGRE',fontname='cmr10',fontsize=16)
plt.colorbar()
plt.savefig("Ex1_Sinogram_TIGRE.pdf", format="pdf", bbox_inches="tight")

# %%

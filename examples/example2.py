''' 
Example 2: How to use the ABBA methods

Author: Maria Knudsen (February 2023)
'''
# %%
# Load packages
import example1 as ex1
from projector_setup import *
from solvers import *
import matplotlib.pyplot as plt
import numpy as np

# Reference to CT setup
ct = ex1.CT_ASTRA

# Create noisy sinogram
rnl     = 0.03
e0      = np.random.normal(0.0, 1.0, ct.m)
e1      = e0/np.linalg.norm(e0)
bexact  = ex1.Bexact.reshape(-1)
e       = rnl*np.linalg.norm(bexact)*e1
b       = bexact + e

# Setup for ABBA methods
A       = fp_astra(ct)     # The forward projector
B       = bp_astra(ct)     # The back projector
iter    = 50               # Maximum number of iterations

# Calls to AB- and BA-GMRES
X_AB, R_AB = AB_GMRES(A,B,b,iter,ct.m, ct.n, ct.num_angles)
X_BA, R_BA = BA_GMRES(A,B,b,iter,ct.m, ct.n, ct.num_angles)



# ===============================================================================================
#  Plotting of results - Example has ended
# ===============================================================================================
# Computing the relative error between the solutions x_i and the true solution
res_AB = np.zeros((iter,1))
res_BA = np.zeros((iter,1))
for i in range(0,iter):
    res_AB[i] = np.linalg.norm(ex1.X.reshape(-1) - X_AB[:,i])/np.linalg.norm(ex1.X.reshape(-1))
    res_BA[i] = np.linalg.norm(ex1.X.reshape(-1) - X_BA[:,i])/np.linalg.norm(ex1.X.reshape(-1))
val_AB = np.min(res_AB)
val_BA = np.min(res_BA)
idx_AB = np.argmin(res_AB)
idx_BA = np.argmin(res_BA)

# Plotting
plt.figure()
plt.plot(range(0,iter),res_AB,'r-')
plt.plot(range(0,iter),res_BA,'k-')
plt.plot(idx_AB,val_AB,'r*')
plt.plot(idx_BA,val_BA,'k*')
plt.title('Convergence History',fontname='cmr10',fontsize=16)
plt.xlabel('Iteration',fontname='cmr10',fontsize=16)
plt.ylabel('Relative error',fontname='cmr10',fontsize=16)
plt.legend(['AB-GMRES','BA-GMRES',
            'iter ='+str(idx_AB)+', error = '+str(round(val_AB,4)),
            'iter ='+str(idx_BA)+', error = '+str(round(val_BA,4))])
plt.savefig("Ex2_convergence.pdf", format="pdf", bbox_inches="tight")

xmin = np.min((np.min(ex1.X,),np.min(X_AB[:,idx_AB]),np.min(X_BA[:,idx_BA])))
xmax = np.max((np.max(ex1.X,),np.max(X_AB[:,idx_AB]),np.max(X_BA[:,idx_BA])))
num_pixels = ex1.num_pixels
fig, axs = plt.subplots(1,3, figsize=(16,4))
im0 = axs[0].imshow(ex1.X,vmin=xmin,vmax=xmax)
axs[0].set_title("Exact Image",fontname='cmr10',fontsize=16)
plt.colorbar(im0, ax=axs[0])
im1 = axs[1].imshow(X_AB[:,idx_AB].reshape(num_pixels,num_pixels),vmin=xmin,vmax=xmax)
axs[1].set_title("AB-GMRES",fontname='cmr10',fontsize=16)
plt.colorbar(im1, ax=axs[1])
im2 = axs[2].imshow(X_BA[:,idx_BA].reshape(num_pixels,num_pixels),vmin=xmin,vmax=xmax)
axs[2].set_title("BA-GMRES",fontname='cmr10',fontsize=16)
plt.colorbar(im2, ax=axs[2])
plt.savefig("Ex2_optimal_recons.pdf", format="pdf", bbox_inches="tight")

# %%

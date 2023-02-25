''' 
Example 3: How to use restart in the ABBA methods, we consider BA-GMRES here but the same applies to AB-GMRES.  
        - We consider a case where the restart parameter p is a divisor of the maximum number of iterations and 
          one where it is not a divisor. Thus, we illustrate that the ABBA methods can handle problems where the 
          restart parameter is not a divisor of the maximum number of iterations. The ABBA methods will take the highest 
          number of iterations without exceeding the maximum number of iterations.

Author: Maria Knudsen (February 2023)
'''
# %%
# Load packages
import example1 as ex1
from projector_setup import *
from solvers import *
import matplotlib.pyplot as plt
import numpy as np

# Create noisy sinogram
rnl     = 0.03
e0      = np.random.normal(0.0, 1.0, ex1.CT_ASTRA.m)
e1      = e0/np.linalg.norm(e0)
bexact  = ex1.Bexact.reshape(-1)
e       = rnl*np.linalg.norm(bexact)*e1
b       = bexact + e

# Setup for ABBA methods
A           = fp_astra(ex1.CT_ASTRA)                         # The forward projector
B           = bp_astra(ex1.CT_ASTRA)                         # The back projector
iter        = 100                                            # Maximum number of iterations

# Use restart with p as a multiplum of the iterations
p = 5       # Restart parameter, if p = iter we do not use restart. 
X_BA_p5, R_BA_p5 = BA_GMRES(A,B,b,iter,ex1.CT_ASTRA,p)     # Solving the CT problem with BA-GMRES for p = 5

# Computing the relative error between the solutions x_i and the true solution
res_p5 = np.zeros((iter,1))
for i in range(0,iter):
    res_p5[i] = np.linalg.norm(ex1.X.reshape(-1) - X_BA_p5[:,i])/np.linalg.norm(ex1.X.reshape(-1))
val_p5 = np.min(res_p5)
idx_p5 = np.argmin(res_p5)

# Use restart with p not being a multiplum of the iterations
p = 6       # Restart parameter, if p = iter or is not included, we do not use restart. 
X_BA_p6, R_BA_p6 = BA_GMRES(A,B,b,iter,ex1.CT_ASTRA,p)     # Solving the CT problem with BA-GMRES for p = 6

# Computing the relative error between the solutions x_i and the true solution
iter_p6 = np.shape(X_BA_p6)[1] - 1
res_p6 = np.zeros((iter_p6,1))
for i in range(0,iter_p6):
    res_p6[i] = np.linalg.norm(ex1.X.reshape(-1) - X_BA_p6[:,i])/np.linalg.norm(ex1.X.reshape(-1))
val_p6 = np.min(res_p6)
idx_p6 = np.argmin(res_p6)

plt.figure()
plt.plot(range(0,iter),res_p5,'r-')
plt.plot(range(0,iter_p6),res_p6,'k-')
plt.plot(idx_p5,val_p5,'r*')
plt.plot(idx_p6,val_p6,'k*')
plt.title('Convergence History',fontname='cmr10',fontsize=16)
plt.xlabel('Iteration',fontname='cmr10',fontsize=16)
plt.ylabel('Relative error',fontname='cmr10',fontsize=16)
plt.legend(['BA-GMRES(5)','BA-GMRES(6)',
            'iter ='+str(idx_p5)+', error = '+str(round(val_p5,4)),
            'iter ='+str(idx_p6)+', error = '+str(round(val_p6,4))])
plt.savefig("Ex3_convergence.pdf", format="pdf", bbox_inches="tight")

num_pixels = ex1.num_pixels
fig, axs = plt.subplots(1,3, figsize=(16,4))
im0 = axs[0].imshow(ex1.X)
axs[0].set_title("Exact Image",fontname='cmr10',fontsize=16)
plt.colorbar(im0, ax=axs[0])
im1 = axs[1].imshow(X_BA_p5[:,idx_p5].reshape(num_pixels,num_pixels))
axs[1].set_title("BA-GMRES(5)",fontname='cmr10',fontsize=16)
plt.colorbar(im1, ax=axs[1])
im2 = axs[2].imshow(X_BA_p6[:,idx_p6].reshape(num_pixels,num_pixels))
axs[2].set_title("BA-GMRES(6)",fontname='cmr10',fontsize=16)
plt.colorbar(im2, ax=axs[2])
plt.savefig("Ex3_optimal_recons.pdf", format="pdf", bbox_inches="tight")

# %%

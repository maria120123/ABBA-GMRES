''' 
Example 4: How to use the stopping rules in the ABBA methods. 
           - We consider BA-GMRES here but the same applies to AB-GMRES. 

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
x0          = np.zeros((ex1.CT_ASTRA.n,)).astype("float32")  # Initial guess of the solution
iter        = 100                                            # Maximum number of iterations
p           = iter                                           # Restart parameter, if p = iter we do not use restart

# Using no stopping rule
X_BA, R_BA = BA_GMRES(A,B,b,x0,iter,ex1.CT_ASTRA,p)     # Solving the CT problem with BA-GMRES

# Computing the relative error between the solutions x_i and the true solution
res = np.zeros((iter,1))
for i in range(0,iter):
    res[i] = np.linalg.norm(ex1.X.reshape(-1) - X_BA[:,i])/np.linalg.norm(ex1.X.reshape(-1))
val = np.min(res)
idx = np.argmin(res)

# Using DP as the stopping rule
stop_rule   = 'DP'          # The stopping criteria, DP = discrepancy principal
eta         = np.std(e)     # The noise level, when using DP as stopping criteria
tau         = 1.02          # Safety factor for DP
X_BA_dp, R_BA_dp = BA_GMRES(A,B,b,x0,iter,ex1.CT_ASTRA,p,stop_rule,eta,tau)     # Solving the CT problem with BA-GMRES

# Computing the relative error between the solutions x_i and the true solution
iter_dp = np.shape(X_BA_dp)[1] - 1
res_dp = np.zeros((iter_dp,1))
for i in range(0,iter_dp):
    res_dp[i] = np.linalg.norm(ex1.X.reshape(-1) - X_BA_dp[:,i])/np.linalg.norm(ex1.X.reshape(-1))
val_dp = np.min(res_dp)
idx_dp = np.argmin(res_dp)

# Using DP as the stopping rule
stop_rule   = 'NCP'      # The stopping criteria, NCP = normalized cumulative periodogram
X_BA_ncp, R_BA_ncp = BA_GMRES(A,B,b,x0,iter,ex1.CT_ASTRA,p,stop_rule,eta,tau)     # Solving the CT problem with BA-GMRES

# Computing the relative error between the solutions x_i and the true solution
iter_ncp = np.shape(X_BA_ncp)[1] - 1
res_ncp = np.zeros((iter_ncp,1))
for i in range(0,iter_ncp):
    res_ncp[i] = np.linalg.norm(ex1.X.reshape(-1) - X_BA_ncp[:,i])/np.linalg.norm(ex1.X.reshape(-1))
val_ncp = np.min(res_ncp)
idx_ncp = np.argmin(res_ncp)

plt.figure()
plt.plot(range(0,iter),res,'g-')
plt.plot(range(0,iter_dp),res_dp,'r-')
plt.plot(range(0,iter_ncp),res_ncp,'k-')
plt.plot(idx,val,'g*')
plt.plot(idx_dp,val_dp,'r*')
plt.plot(idx_ncp,val_ncp,'k*')
plt.title('Convergence History',fontname='cmr10',fontsize=16)
plt.xlabel('Iteration',fontname='cmr10',fontsize=16)
plt.ylabel('Relative error',fontname='cmr10',fontsize=16)
plt.legend(['BA-GMRES, No Stopping Rule','BA-GMRES, DP','BA-GMRES, NCP',
            'iter ='+str(idx)+', error = '+str(round(val,4)),
            'iter ='+str(idx_dp)+', error = '+str(round(val_dp,4)),
            'iter ='+str(idx_ncp)+', error = '+str(round(val_ncp,4))])
plt.savefig("Ex4_convergence.pdf", format="pdf", bbox_inches="tight")

num_pixels = ex1.num_pixels
fig, axs = plt.subplots(2,2, figsize=(10,10))
im00 = axs[0,0].imshow(ex1.X)
axs[0,0].set_title("Exact Image",fontname='cmr10',fontsize=16)
axs[0,0].axis('off')
plt.colorbar(im00, ax=axs[0,0])
im01 = axs[0,1].imshow(X_BA[:,idx].reshape(num_pixels,num_pixels))
axs[0,1].set_title("BA-GMRES($\infty$), No SR",fontname='cmr10',fontsize=16)
axs[0,1].axis('off')
plt.colorbar(im01, ax=axs[0,1])
im10 = axs[1,0].imshow(X_BA_dp[:,idx_dp].reshape(num_pixels,num_pixels))
axs[1,0].set_title("BA-GMRES($\infty$), SR = DP",fontname='cmr10',fontsize=16)
axs[1,0].axis('off')
plt.colorbar(im10, ax=axs[1,0])
im11 = axs[1,1].imshow(X_BA_ncp[:,idx_ncp].reshape(num_pixels,num_pixels))
axs[1,1].set_title("BA-GMRES($\infty$), SR = NCP",fontname='cmr10',fontsize=16)
axs[1,1].axis('off')
plt.colorbar(im11, ax=axs[1,1])
plt.savefig("Ex4_optimal_recons.pdf", format="pdf", bbox_inches="tight")

# %%

''' 

Example 3: How to use restart in the ABBA methods, we consider BA-GMRES here but the same applies to AB-GMRES. 
           - We both consider a case where the restart parameter p is a multiplum of the maximum number of iterations and
             one where it is not a multiplum. Thus, we illustrate that the ABBA methods can handle such problems. The solution
             is that the ABBA methods will take the highest number of iterations without exceeding the maximum number of iterations.

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
eta         = np.std(e)                                      # The noise level, when using DP as stopping criteria
stop_rule   = 'NO'                                           # The stopping criteria ('NO' just means we do not use any stopping rule here)

# Use restart with p as a multiplum of the iterations
p = 5       # Restart parameter, if p = iter we do not use restart. 
X_BA_p5, R_BA_p5, T_BA_p5 = BA_GMRES(A,B,b,x0,iter,ex1.CT_ASTRA,p,eta,stop_rule)     # Solving the CT problem with BA-GMRES for p = 5

# Computing the relative error between the solutions x_i and the true solution
res_p5 = np.zeros((iter,1))
for i in range(0,iter):
    res_p5[i] = np.linalg.norm(ex1.X.reshape(-1) - X_BA_p5[:,i])/np.linalg.norm(ex1.X.reshape(-1))
val_p5 = np.min(res_p5)
idx_p5 = np.argmin(res_p5)

# Use restart with p not being a multiplum of the iterations
p = 6       # Restart parameter, if p = iter we do not use restart. 
X_BA_p6, R_BA_p6, T_BA_p6 = BA_GMRES(A,B,b,x0,iter,ex1.CT_ASTRA,p,eta,stop_rule)     # Solving the CT problem with BA-GMRES for p = 6

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
plt.title('Convergence history')
plt.xlabel('Iterations')
plt.ylabel('Relative error |x* - x|_2/|x*|_2')
plt.legend(['BA-GMRES(5)','BA-GMRES(6)',
            'iter ='+str(idx_p5)+', error = '+str(round(val_p5,4)),
            'iter ='+str(idx_p6)+', error = '+str(round(val_p6,4))])
plt.show()
plt.savefig("Ex3_convergence.pdf", format="pdf", bbox_inches="tight")

num_pixels = ex1.num_pixels
fig, axs = plt.subplots(1,3, figsize=(14,10))
axs[0].imshow(ex1.X)
axs[0].set_title("Ground truth")
axs[1].imshow(X_BA_p5[:,idx_p5].reshape(num_pixels,num_pixels))
axs[1].set_title("Best Reconstruction from BA-GMRES(5)")
axs[2].imshow(X_BA_p6[:,idx_p6].reshape(num_pixels,num_pixels))
axs[2].set_title("Best Reconstruction from BA-GMRES(6)")
plt.savefig("Ex3_optimal_recons.pdf", format="pdf", bbox_inches="tight")

# %%

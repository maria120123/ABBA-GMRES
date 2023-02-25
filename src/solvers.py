# %% ***************************************************** Import Packages *****************************************************
import numpy as np
import time

# %% *********************************************** The ABBA Iterative Methods ***********************************************
def AB_GMRES(A, B, b, iter, ct, p = 0, stop_rule = 'NO', eta = 0, tau = 1.02, x0 = 0):
    ''' Solver: AB-GMRES
    Solves min || b - A*B*y ||_2 for y in R^[m] where x = B*y with B in R^[n x m] as a right preconditioner.
    
    Cite: Hansen et al., "GMRES methods for tomographic reconstruction with an unmatched back projector"
    
    INPUT
    A:    Forward projection (can both be a dense and abstract matrix)
    B:    Back projection (can both be a dense and abstract matrix)
    b:    The sinogram
    x0:   Initial guess
    iter: Maximum number of iterations the solver must take.
    ct:   The class describing the CT problem
    p:    Restart parameter, number of iterations before restart
    eta:  Relative noise level
    tau:  Safety factor in DP
    stop_rule: 'DP' for Discrepancy Principle and 'NCP' for Normalized Cumulative Periodogram
    
    OUTPUT
    X: Solution matrix, k'th column is the solution to the k'th iteration
    R: Residual matrix, k'th column corresponds to the residuals for the k'th iteration 
    
    '''
    print("\nAB-GMRES is running")

    # Check if GMRES should be restarted
    if p == 0:
        p = iter

    # Check if a starting guess was provided
    if ~isinstance(x0, np.ndarray):
        x0 = np.zeros((ct.n,)).astype("float32")
    
    # Make sure p is a divisor of iter else change iter
    L = np.floor(iter/p).astype(int)
    if np.mod(iter,p) != 0:
        iter = L*p

    # Initializations
    m = ct.m
    n = ct.n

    X = np.zeros((n,iter+1), dtype='float32')
    X[:,0] = x0
    Xp = np.zeros((n,p), dtype='float32')
    R = np.zeros((m,iter), dtype='float32')

    r0 = b - A @ x0
    for l in range(0,L):
        beta   = np.linalg.norm(r0) 

        W = np.zeros((m,p+1), dtype='float32')    
        W[:,0] = r0/beta # Initialization of the first Krylov subspace vector
        
        # Construct the next Krylov subspace vector and solve the least squares problem
        for k in range(1,p+1):
            print("iteration", str(l*p + k), "out of",str(iter),end="\r")
            
            H = np.zeros((k+1,k),dtype='float32') # Initialize/expand the Hessenberg matrix
            
            # Insert the previous values of the Hessenberg matrix
            if k > 1:
                H[:k,:k-1] = h_old

            q = A @ (B @ W[:,k-1])
            e = np.zeros((k+1,), dtype='float32')
            e[0] = 1

            # Schmidt orthogonalizing the Krylov subspace vector (modified Gram-Schmidt)
            for i in range(1,k+1):
                H[i-1,k-1] = q.reshape(m,1).T @ W[:,i-1].reshape(m,1)
                q = q - H[i-1,k-1]*W[:,i-1] 
            H[k,k-1] = np.linalg.norm(q)
            W[:,k] = q/H[k,k-1] 
            
            # Solve the least squares problem
            y = np.linalg.lstsq(H,(beta*e).reshape(k+1,1),rcond = None)[0]

            # The solution x_k and its residual
            Xp[:,k-1] = x0 + (B @ (W[:,:k] @ np.float32(y))).reshape(-1)
            R[:,k-1] = b - A @ Xp[:,k-1]
            h_old = H
        
            # Stopping rule goes here
            if stop_rule == 'DP':
                if np.linalg.norm(R[:,k-1]) <= tau*eta*np.sqrt(m):
                    X[:,l*p+1:l*p+k+1] = Xp[:,:k]
                    X = X[:,:l*p + k+1]
                    R = R[:,:l*p + k]
                    return X, R
            
            elif stop_rule == 'NCP':
                Nk = NCP(R[:,k-1],ct)
                if l == 0 and k == 1:
                    Nk_old = Nk
                else:
                    #print('different: ',(Nk_old - Nk))
                    if (Nk_old - Nk) < 0:
                        X[:,l*p+1:l*p+k+1] = Xp[:,:k]
                        X = X[:,:l*p + k+1]
                        R = R[:,:l*p + k]
                        return X, R
                    else:
                        Nk_old = Nk

        x0 = Xp[:,k-1]
        r0 = R[:,k-1]
        X[:,l*p+1:l*p+k+1] = Xp

    return X, R

def BA_GMRES(A, B, b, iter, ct, p = 0, stop_rule = 'NO', eta = 0, tau = 1.02, x0 = 0):
    ''' Solver: BA-GMRES
    Solves min || B*b - B*A*x ||_2  with B in R^[n x m] as a left preconditioner.
    
    Cite: Hansen et al., "GMRES methods for tomographic reconstruction with an unmatched back projector"
    
    INPUT
    A:    Forward projection (can both be a dense and abstract matrix)
    B:    Back projection (can both be a dense and abstract matrix)
    b:    The sinogram/measurements
    x0:   Initial guess
    iter: Maximum number of iterations the solver must take.
    ct:   The class describing the CT problem
    p:    Restart parameter, number of iterations before restart
    eta:  Relative noise level
    tau:  Safety factor in DP
    stop_rule: 'DP' for Discrepancy Principle and 'NCP' for Normalized Cumulative Periodogram
    
    OUTPUT
    X: Solution matrix, k'th column is the solution to the k'th iteration
    R: Residual matrix, k'th column corresponds to the residuals for the k'th iteration
    
    '''
    print("\nBA-GMRES is running")
    
    # Check if GMRES should be restarted
    if p == 0:
        p = iter

    # Check if a starting guess was provided
    if ~isinstance(x0, np.ndarray):
        x0 = np.zeros((ct.n,)).astype("float32")

    # Make sure p is a divisor of iter else change iter
    L = np.floor(iter/p).astype(int)
    if np.mod(iter,p) != 0:
        iter = L*p

    # Initializations
    m = ct.m
    n = ct.n
    b = np.float32(b)

    X = np.zeros((n,iter+1), dtype='float32')
    X[:,0] = x0
    Xp = np.zeros((n,p), dtype='float32')
    R = np.zeros((m,iter), dtype='float32')
    
    residual = b - A @ x0
    for l in range(0,L):
        r0 = B @ (residual)
        beta = np.linalg.norm(r0)

        W = np.zeros((n,p+1), dtype='float32')
        W[:,0] = r0/beta # Initialization of the first Krylov subspace vector
        
        # Construct the next Krylov subspace vector and solve the least squares problem
        for k in range(1,p+1):
            print("iteration", str(l*p + k), "out of",str(iter),end="\r")

            H = np.zeros((k+1,k), dtype='float32') # Initialize/expand the Hessenberg matrix

            # Insert the previous values of the Hessenberg matrix
            if k > 1:
                H[:k,:k-1] = h_old
            
            q = B @ (A @ W[:,k-1])
            e = np.zeros((k+1,), dtype='float32')
            e[0] = 1

            # Schmidt orthogonalizing the Krylov subspace vector (modified Gram-Schmidt)
            for i in range(1,k+1):
                H[i-1,k-1] = q.reshape(n,1).T @ W[:,i-1].reshape(n,1)
                q = q - H[i-1,k-1]*W[:,i-1] 
            H[k,k-1] = np.linalg.norm(q)
            W[:,k] = q/H[k,k-1] 
            
            # Solve the least squares problem
            y = np.linalg.lstsq(H,(beta*e).reshape(k+1,1),rcond = None)[0]

            # The solution x_k and its residual
            Xp[:,k-1] = x0 + (W[:,:k] @ np.float32(y)).reshape(-1)
            R[:,k-1] = b - A @ Xp[:,k-1]
            h_old = H
            
            # Stopping rule goes here
            if stop_rule == 'DP': 
                if np.linalg.norm(R[:,k-1]) <= tau*eta*np.sqrt(m):
                    X[:,l*p+1:l*p+k+1] = Xp[:,:k]
                    X = X[:,:l*p + k+1]
                    R = R[:,:l*p + k]
                    return X, R
            
            elif stop_rule == 'NCP':
                Nk = NCP(R[:,k-1],ct)
                if l == 0 and k == 1:
                    Nk_old = Nk
                else:
                    if (Nk_old - Nk) < 0:
                        X[:,l*p+1:l*p+k+1] = Xp[:,:k]
                        X = X[:,:l*p + k+1]
                        R = R[:,:l*p + k]
                        return X, R
                    else:
                        Nk_old = Nk

        x0 = Xp[:,k-1]
        residual = R[:,k-1]
        X[:,l*p+1:l*p+k+1] = Xp

    return X, R

def NCP(r,ct):
    ''' 
    Stopping criteria: Normalized Cumulative Periodogram

    INPUT
    r:  Residual vector for i'th iteration
    ct: The class for the CT problem
    '''
    
    nt = int(ct.N_ang)
    nnp = int(ct.m / nt)
    q = int(np.floor(nnp/2))
    c_white = np.linspace(1,q,q)/q
    C = np.zeros((q,nt))
    
    R = r.reshape(nnp,nt)
    for j in range(0,nt):
        RKH = np.fft.fft(R[:,j])
        pk = abs(RKH[0:q+1]) #+1?
        c = np.cumsum(pk[1:])/np.sum(pk[1:])
        C[:,j] = c

    Nk = np.linalg.norm(np.mean(C,1)-c_white)
    return Nk
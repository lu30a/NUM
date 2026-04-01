from numpy import linalg as la
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres, spsolve


def jacobi(a, b):
    n = len(b)
    x = np.zeros(n)
    x1 = np.zeros(n)
    iter = 0
    #maxiter = n**4
    maxiter=n*40
    err = np.inf
    while err > np.finfo(float).eps and iter < maxiter:
        iter += 1
        for i in range(n):
            sl = a[i, :i] @ x[:i]   if i > 0     else 0.0
            su = a[i, i+1:] @ x[i+1:] if i < n-1 else 0.0
            x1[i] = (b[i] - sl - su) / a[i, i]
        err = np.linalg.norm(x1 - x)
        x = x1.copy()
    return x, iter


def gseidel(a, b):
    n = len(b)
    x = np.zeros(n)
    iter_ = 0
    #maxiter = n**4
    maxiter=n*40
    err = np.inf
    while err > np.finfo(float).eps and iter_ < maxiter:
        iter_ += 1
        err = 0.0
        for i in range(n):
            sl = a[i, :i]   @ x[:i]   if i > 0     else 0.0
            su = a[i, i+1:] @ x[i+1:] if i < n - 1 else 0.0
            temp = (b[i] - sl - su) / a[i, i]
            err += (temp - x[i]) ** 2
            x[i] = temp         
        err = np.sqrt(err)
    return x, iter_



def sor(A,b,omega):
    n,m=np.shape(A)
    #maxit=n**2
    maxit=n*20
    tol=1e-12
    x=np.zeros(np.size(b))
    z=x
    y=x
    err=la.norm(b)
    iter=0
    continua=(err>tol)&(iter<maxit)
    while continua:
        iter+=1
        err=0
        for i in range(0,n):
            sl=0
            su=0
            if i>0:
                sl=A[i,0:i]@x[0:i]
            if i<n:
                su=A[i,i+1:n]@x[i+1:n]
            e1=(1-omega)*x[i]+omega*((b[i]-sl-su)/A[i,i])
            err=err+np.abs(e1-x[i])**2
            x[i]=e1
        err=np.sqrt(err)
        continua=(err > tol) & (iter < maxit)
    return x, iter, err

def newton_solver(x0, F, J, rel_tol=1e-8, abs_tol_F=1e-10, maxiter=50, lin_rtol=1e-8, verbose=False):

    x = np.array(x0, dtype=float)

    for niter in range(maxiter):
        Fx = F(x)
        Fnorm = np.linalg.norm(Fx)

        if Fnorm < abs_tol_F:
            if verbose:
                print(f"Newton converged: ||F|| = {Fnorm:.3e} at iter {niter}")
            return x, niter


        Jx = J(x)
      
        dx, info = gmres(Jx, -Fx, rtol=lin_rtol)
        
        
        if info != 0:
            if verbose:
                print(f"newton solver -> gmres info = {info}")
            break

        alpha, niter2 = damping(x, dx, F, gamma=0.9, maxit=10)
        niter += niter2 + 1

        x_new = x + alpha * dx
        
        step_norm = np.linalg.norm(dx)
        xnorm = max(1.0, np.linalg.norm(x_new))
        rel_step = step_norm / xnorm

        if rel_step < rel_tol:
            if verbose:
                print(f"Newton converged: rel_step = {rel_step:.3e} at iter {niter+1}")
            return x_new, niter + 1

        x = x_new

    return x, maxiter

from scipy.sparse import lil_matrix

def assemble_fd_jacobian(F, x, h=1e-6):
    x = np.array(x, dtype=float)
    n = x.size
    Fx = F(x)
    
    J = lil_matrix((n, n))
    
    for j in range(n):
        xj = x.copy()
        xj[j] += h
        Fj = F(xj)
        J[:, j] = (Fj - Fx) / h
    
    return J.tocsr() 


def quasi_newton_solver(x0, F, rel_tol=1e-8, maxiter=50, lin_rtol=1e-6, verbose=False):
    x = np.array(x0, dtype=float).reshape(-1)
    n = x.size
    niter = 0

    while niter < maxiter:
        Fx = np.array(F(x), dtype=float).reshape(-1)
        Fnorm = np.linalg.norm(Fx)

        if Fnorm == 0:
            return x, niter

        Jx = assemble_fd_jacobian(F, x, h=1e-6)

        dx, info = gmres(Jx, -Fx, rtol=lin_rtol)

        if info != 0:
            if verbose:
                print(f"quasi-newton solver -> info = {info}")
            break

        alpha, niter2 = damping(x, dx, F, gamma=0.6, maxit=20)
        niter += niter2 + 1

        x_new = x + alpha * dx

        rel_error = np.linalg.norm(dx) / max(np.linalg.norm(x_new), 1.0)
        
        if rel_error < rel_tol:
            return x_new, niter

        x = x_new

    return x, niter


def damping(x, s, F, gamma=0.5, maxit=20):
    alpha = 1.0
    niter = 0
    Fx = F(x).reshape(-1)

    while niter < maxit:
        v = x + alpha * s
        if np.linalg.norm(F(v).reshape(-1)) < gamma * np.linalg.norm(Fx):
            return alpha, niter
        alpha *= 0.5
        niter += 1

    return alpha, niter

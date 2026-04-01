import numpy as np

def newton_solver(x0, F, J, rel_tol, maxiter):
    x = np.array(x0, dtype=float)
    for niter in range(maxiter):
        Fx = F(x)
        Jx = J(x)
        try:
            dx = np.linalg.solve(Jx, Fx)
        except np.linalg.LinAlgError:
            dx = np.linalg.pinv(Jx) @ Fx

        x_new = x - dx
        if np.linalg.norm(x_new) > 0:
            rel_error = np.linalg.norm(dx) / np.linalg.norm(x_new)
        else:
            rel_error = np.linalg.norm(dx)
        if rel_error < rel_tol:
            return x_new, niter + 1
        x = x_new
    print("Warning: Newton solver did not converge within the maximum number of iterations")
    return x, maxiter


def quasi_newton_solver(x0, F, rel_tol, maxiter):
    h = 1e-6
    x = np.array(x0, dtype=float)
    n = len(x)
    niter = 0
    while niter < maxiter:
        Jx = np.zeros((n, n))
        Fx = F(x)
        for i in range(n):
            x_h = x.copy()
            x_h[i] += h
            Fx_h = F(x_h)
            Jx[:, i] = (Fx_h - Fx) / h

        try:
            dx = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            dx = np.linalg.pinv(Jx) @ (-Fx)

        alpha, niter2 = damping(x, dx, F, gamma=0.5, maxit=20)
        niter += niter2 + 1
        x_new = x + alpha * dx

        rel_error = np.linalg.norm(dx) / np.linalg.norm(x_new) if np.linalg.norm(x_new) > 0 else np.linalg.norm(dx)
        if rel_error < rel_tol:
            return x_new, niter
        x = x_new

    print("Warning: Quasi-Newton solver did not converge within the maximum number of iterations")
    return x, niter

def damping(x, s, F, gamma=0.5, maxit=20):
    alpha = 1.0
    niter = 0
    while niter < maxit:
        v = x + alpha * s
        if np.linalg.norm(F(v)) < gamma * np.linalg.norm(F(x)):
            return alpha, niter
        alpha *= 0.5
        niter += 1
    return alpha, niter
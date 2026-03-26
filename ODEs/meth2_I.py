import numpy as np
from quasi_nwt import newton_solver, quasi_newton_solver

def impl_euler(f, tspan, y0, N_steps, J=None, tau=1e-19, maxit=None):
    t0, tf = tspan[0], tspan[1]
    h = (tf - t0) / (N_steps - 1)
    t = np.linspace(tspan[0], tspan[1], N_steps)
    y0 = np.atleast_1d(y0).astype(float)
    m = y0.size
    u = np.zeros((m, N_steps))
    u[:, 0] = y0
    if maxit is None:
        maxit = int(min(1e4, 100 * m))
    feval = 0

    for k in range(1, N_steps):
        tk = t[k]
        u_prev = u[:, k-1].copy()
        def F(x):
            return x - h * f(tk, x) - u_prev
        if J is not None:
            JJ = J(t[k-1], u_prev)
            JJ_arr = np.asarray(JJ)
            if JJ_arr.ndim == 2:
                JJ_mat = JJ_arr
            elif JJ_arr.size == 1:
                JJ_mat = JJ_arr.item() * np.eye(m)
            else:
                JJ_mat = JJ_arr[-1] * np.eye(m)

            def jac(x):
                return np.eye(m) - h * JJ_mat

            u_k, niter = newton_solver(u_prev, F, jac, tau, maxit)
        else:
            u_k, niter = quasi_newton_solver(u_prev, F, tau, maxit)

        feval += niter
        u[:, k] = np.atleast_1d(u_k)

    return t, u, feval

def impl_euler_quasi(f, tspan, y0, N_steps, tau=1e-5, maxit=None):
    return impl_euler(f, tspan, y0, N_steps, J=None, tau=tau, maxit=maxit)


def impl_CN(f, tspan, y0, N_steps, J=None, tau=1e-5, maxit=None):

    t0, T = tspan
    h = (T - t0) / (N_steps - 1)
    t = np.linspace(t0, T, N_steps)
    y0 = np.asarray(y0)
    n = y0.size
    u = np.zeros((n, t.size))
    u[:, 0] = y0.copy()

    if maxit is None:
        maxit = int(min(1e4, 100 * n))

    feval = 0
    for k in range(1, N_steps):
        tk = t[k]
        tkm1 = t[k - 1]
        u_prev = u[:, k - 1].copy()
        rhs_const = u_prev + (h / 2.0) * f(tkm1, u_prev)

        def F(x):
            return x - rhs_const - (h / 2.0) * f(tk, x)

        if J is not None:
            def jac_F(x):
                Jf = np.asarray(J(tk, x))
                if Jf.ndim == 2:
                    Jmat = Jf
                elif Jf.size == 1:
                    Jmat = Jf.item() * np.eye(n)
                else:
                    Jmat = Jf[-1] * np.eye(n)
                return np.eye(n) - (h / 2.0) * Jmat

            u_k, niter = newton_solver(u_prev, F, jac_F, tau, maxit)
            feval += niter
        else:
            u_k, niter = quasi_newton_solver(u_prev, F, tau, maxit)
            feval += niter

        u[:, k] = u_k

    return t, u, feval

def impl_CN_quasi(f, tspan, y0, N_steps, tau=1e-5, maxit=None):
    return impl_CN(f, tspan, y0, N_steps, J=None, tau=tau, maxit=maxit)

def sys(butcher, KK, n, s, TT, u, h, f):
    KK = np.asarray(KK).ravel()
    PP = np.zeros((n, s))
    for j in range(s):
        KKj = KK[j*n:(j+1)*n]
        for i in range(s):
            PP[:, i] += butcher['A'][i, j] * KKj
    Fvec = np.zeros(s * n)
    for i in range(s):
        Fvec[i*n:(i+1)*n] = KK[i*n:(i+1)*n] - f(TT[i], u + h * PP[:, i])
    return Fvec

def jac(butcher, KK, n, s, TT, u, h, fy):
    KK = np.asarray(KK).ravel()
    PP = np.zeros((n, s))
    for j in range(s):
        KKj = KK[j*n:(j+1)*n]
        for i in range(s):
            PP[:, i] += butcher['A'][i, j] * KKj
    D = [fy(TT[i], u + h * PP[:, i]) for i in range(s)]  # each should be n x n
    J = np.zeros((s*n, s*n))
    for i in range(s):
        for j in range(s):
            J[i*n:(i+1)*n, j*n:(j+1)*n] = -h * butcher['A'][i, j] * D[i]
        J[i*n:(i+1)*n, i*n:(i+1)*n] += np.eye(n)
    return J

def K_solve(butcher, u, f, fy, t, T, method):
    s = len(butcher['c'])
    n = u.size
    h = T - t
    maxiter = 150
    tau = 1e-6
    n_it = 0
    TT = t + h * np.array(butcher['c'], dtype=float)
    F = lambda KK: sys(butcher, KK, n, s, TT, u, h, f)
    K0 = np.hstack([f(TT[i], u) for i in range(s)])
    Jfun = lambda KK: jac(butcher, KK, n, s, TT, u, h, fy)
    if method == "Nwt":
        K, niter = newton_solver(K0, F, Jfun, tau, maxiter)
        n_it += niter
        return K, n_it
    if method == "QuasiNwt":
        K, niter = quasi_newton_solver(K0, F, tau, maxiter)
        n_it += niter
        return K, n_it
    return K0, n_it

def IRK3_solver(f, J, u0, Ti, N, meth1, meth2, Ntime=None):
    if meth1 == 'GLe':
        b = np.array([5/18, 4/9, 5/18], dtype=float)
        c = np.array([0.5 - (np.sqrt(15)/10), 0.5, 0.5 + (np.sqrt(15)/10)], dtype=float)
        sqrt15 = np.sqrt(15)
        A = np.array([
            [5/36, 2/9 - sqrt15/15, 5/36 - sqrt15/30],
            [5/36 + sqrt15/24, 2/9, 5/36 - sqrt15/24],
            [5/36 + sqrt15/30, 2/9 + sqrt15/15, 5/36]
        ], dtype=float)
    elif meth1 == 'GRa':
        b = np.array([(16-np.sqrt(6))/36, (16+np.sqrt(6))/36, 1/9], dtype=float)
        c = np.array([0.5 - np.sqrt(6)/10, 0.5 + np.sqrt(6)/10, 1.0], dtype=float)
        A = np.array([
            [(88-7*np.sqrt(6))/360, (296-169*np.sqrt(6))/1800, (-2+3*np.sqrt(6))/225],
            [(296+169*np.sqrt(6))/1800, (88+7*np.sqrt(6))/360, (-2-3*np.sqrt(6))/225],
            [(16-np.sqrt(6))/36, (16+np.sqrt(6))/36, 1/9]
        ], dtype=float)
    elif meth1 == 'GLo':
        b = np.array([1/6, 2/3, 1/6], dtype=float)
        c = np.array([0.0, 0.5, 1.0], dtype=float)
        A = np.array([[0.0, 0.0, 0.0],
                      [1/4, 1/4, 0.0],
                      [0.0, 1.0, 0.0]], dtype=float)
    else:
        raise ValueError("Unknown IRK method: %s" % meth1)

    butcher = {'b': b, 'c': c, 'A': A}

    tgrid = np.linspace(Ti[0], Ti[1], N)

    n = u0.size
    u = np.zeros((n, N))
    u[:, 0] = u0
    num_fval = 0
    s = len(b)
    for k in range(1, N):
        h = tgrid[k] - tgrid[k-1]
        K, n_it = K_solve(butcher, u[:, k-1], f, J, tgrid[k-1], tgrid[k], meth2)
        Z2 = np.zeros(n)
        for i in range(s):
            Z2 += butcher['b'][i] * K[i*n:(i+1)*n]
        u[:, k] = u[:, k-1] + h * Z2
        num_fval += n_it
    return tgrid, u, num_fval

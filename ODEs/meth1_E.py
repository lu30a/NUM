import numpy as np

def Expl_Euler(f, tspan, y0, N_steps):
    
    t0, tf = tspan[0], tspan[1]
    h = (tf - t0) / N_steps
    t = np.linspace(t0, tf, N_steps + 1)
    y0 = np.atleast_1d(y0)
    u = np.zeros((N_steps + 1, len(y0)))
    u[0, :] = y0
    feval = 0
    for n in range(N_steps):
        f_val = f(t[n], u[n, :])
        u[n + 1, :] = u[n, :] + h * f_val
        feval += 1
    
    return t, u, feval


def Expl_Heun(f, tspan, y0, N_steps):

    t0, tf = tspan
    h = (tf - t0) / N_steps
    t = np.linspace(t0, tf, N_steps + 1)
    y0 = np.atleast_1d(y0).astype(float)
    m = y0.size
    u = np.zeros((N_steps + 1, m))
    u[0, :] = y0
    feval = 0
    for n in range(N_steps):
        tn = t[n]
        yn = u[n, :]
        f_val_n = np.atleast_1d(f(tn, yn))
        u_predictor = yn + h * f_val_n
        f_val_predictor = np.atleast_1d(f(t[n + 1], u_predictor))
        u[n + 1, :] = yn + (h / 2.0) * (f_val_n + f_val_predictor)
        feval += 2

    return t, u, feval


def ERK3(f, tspan, y0, N_steps):

    t0, tf = tspan
    h = (tf - t0) / N_steps
    t = np.linspace(t0, tf, N_steps + 1)
    y0_arr = np.atleast_1d(y0).astype(float)
    m = y0_arr.size
    sol = np.zeros((N_steps + 1, m))
    sol[0] = y0_arr
    A21 = 2.0/3.0
    A31 = 1.0/6.0
    A32 = 1.0/2.0
    b1, b2, b3 = 1.0/4.0, 1.0/4.0, 1.0/2.0
    num_fval = 0
    for n in range(N_steps):
        tn = t[n]
        yn = sol[n]
        k1 = np.atleast_1d(f(tn, yn)); num_fval += 1
        k2 = np.atleast_1d(f(tn + (2.0/3.0)*h, yn + h*A21*k1)); num_fval += 1
        k3 = np.atleast_1d(f(tn + (2.0/3.0)*h, yn + h*(A31*k1 + A32*k2))); num_fval += 1
        sol[n+1] = yn + h*(b1*k1 + b2*k2 + b3*k3)

    if sol.shape[1] == 1:
        sol = sol.reshape(-1, 1)
    return t, sol, num_fval



def ERK4(f, tspan, y0, N):

    c = np.array([0.0, 1/3, 2/3, 1.0])
    A = np.zeros((4, 4))
    A[1, 0] = 1/3
    A[2, 0] = -1/3; A[2, 1] = 1.0
    A[3, 0] = 1.0;  A[3, 1] = -1.0; A[3, 2] = 1.0
    b = np.array([1/8, 3/8, 3/8, 1/8])

    y0_vec = np.atleast_1d(y0).astype(float)
    m = y0_vec.size

    h = (tspan[1] - tspan[0]) / N

    t = np.linspace(tspan[0], tspan[1], N + 1)
    u = np.zeros((N + 1, m))

    u[0, :] = y0_vec

    num_fval = 0
    for k in range(1, N + 1):
        tkm1 = t[k - 1]
        ukm1 = u[k - 1, :]

        K1 = np.atleast_1d(f(tkm1, ukm1)).astype(float)
        K2 = np.atleast_1d(f(tkm1 + c[1] * h, ukm1 + h * A[1, 0] * K1)).astype(float)
        K3 = np.atleast_1d(f(tkm1 + c[2] * h, ukm1 + h * (A[2, 0] * K1 + A[2, 1] * K2))).astype(float)
        K4 = np.atleast_1d(f(tkm1 + c[3] * h, ukm1 + h * (A[3, 0] * K1 + A[3, 1] * K2 + A[3, 2] * K3))).astype(float)

        u[k, :] = ukm1 + h * (b[0] * K1 + b[1] * K2 + b[2] * K3 + b[3] * K4)

        num_fval += 4

    return t, u, num_fval
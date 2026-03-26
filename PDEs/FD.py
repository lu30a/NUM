import numpy as np
from scipy.sparse import diags, eye, kron

def FDlapl_homo(dim,N,hx,M=None,hy=None,L=None,hz=None):
    if dim==1:
        return (1 / hx**2) * (-2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1))
    elif dim==2:
        D_x = (1 / hx**2) * (-2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1))
        D_y = (1 / hy**2) * (-2 * np.eye(M) + np.eye(M, k=1) + np.eye(M, k=-1))
        return np.kron(D_x, np.eye(M)) + np.kron(np.eye(N), D_y)
    elif dim==3:
        D_x = (1 / hx**2) * (-2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1))
        D_y = (1 / hy**2) * (-2 * np.eye(M) + np.eye(M, k=1) + np.eye(M, k=-1))
        D_z = (1 / hz**2) * (-2 * np.eye(L) + np.eye(L, k=1) + np.eye(L, k=-1))
        return np.kron(np.kron(D_x,np.eye(M)),np.eye(L)) + np.kron(np.kron(np.eye(N),D_y), np.eye(L))+np.kron(np.kron(np.eye(N),np.eye(M)), D_z)
    elif dim>3:
        print('not supported yet :(')
        return None

def FDlapl_homo_sparse(dim, N, hx, M=None, hy=None, L=None, hz=None):
    D1 = lambda n, h: (1/h**2) * diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format='csr', dtype=None)
    if dim == 1:
        return D1(N, hx)
    elif dim == 2:
        Dx = D1(N, hx);  Dy = D1(M, hy)
        return kron(Dx, eye(M)) + kron(eye(N), Dy)
    elif dim == 3:
        Dx = D1(N, hx);  Dy = D1(M, hy);  Dz = D1(L, hz)
        return (kron(kron(Dx, eye(M)), eye(L))
              + kron(kron(eye(N), Dy), eye(L))
              + kron(kron(eye(N), eye(M)), Dz))
    
def eval_g(f, *c):
    return f(*c) if callable(f) else float(f)

def applyRobin1D(A, N, h, alpha, beta, id0, id1):
    if alpha[id0] != 0:
        R = beta[id0]/alpha[id0]
        A[0,0] = -2*(1+h*R)/h**2
        A[0,1] = 2/h**2
    if alpha[id1] != 0:
        R = beta[id1]/alpha[id1]
        A[N-1,N-2] = 2/h**2
        A[N-1,N-1] = -2*(1+h*R)/h**2
    return A

def FDLapl_rob_sp_1D(N, h, alpha, beta, id0, id1):
        return applyRobin1D(FDlapl_homo_sparse(1, N, h).tolil(), N, h, alpha, beta, id0, id1).tocsr()


def FDlapl_robin_sparse(dim, N, hx, alpha, beta, M=None, hy=None, L=None, hz=None):
    if dim == 1:
        return FDLapl_rob_sp_1D(N+2, hx, alpha, beta, 'L', 'R')
    if dim == 2: 
        Nx, Ny = N+2, M+2
        Tx = FDLapl_rob_sp_1D(Nx, hx, alpha, beta, 'L', 'R')
        Ty = FDLapl_rob_sp_1D(Ny, hy, alpha, beta, 'B', 'T')
        return (kron(Tx, eye(Ny)) + kron(eye(Nx), Ty)).tocsr()
    if dim == 3:
        Nx, Ny, Nz = N+2, M+2, L+2
        Tx = FDLapl_rob_sp_1D(Nx, hx, alpha, beta, 'L', 'R')
        Ty = FDLapl_rob_sp_1D(Ny, hy, alpha, beta, 'B', 'T')
        Tz = FDLapl_rob_sp_1D(Nz, hz, alpha, beta, 'D', 'U')
        Ix, Iy, Iz = eye(Nx), eye(Ny), eye(Nz)
        return (kron(kron(Tx, Iy), Iz) +
                kron(kron(Ix, Ty), Iz) +
                kron(kron(Ix, Iy), Tz)).tocsr()

def rhs_robin(dim, f_vals, N, hx, alpha, g, xi, M=None, hy=None, yj=None, L=None, hz=None, zk=None):
    if dim == 1:
        r = f_vals.copy().reshape(N+2).astype(float)
        if alpha['L'] != 0: r[0]  -= 2*eval_g(g['L'], xi[0])  / (alpha['L']*hx)
        if alpha['R'] != 0: r[-1] -= 2*eval_g(g['R'], xi[-1]) / (alpha['R']*hx)
        return r
    if dim==2:
        Nx, Ny = N+2, M+2
        r = f_vals.copy().reshape(Nx, Ny).astype(float)
        if alpha['L'] != 0: r[0,  :] -= 2*np.array([eval_g(g['L'], xi[0],   yj[j]) for j in range(Ny)]) / (alpha['L']*hx)
        if alpha['R'] != 0: r[-1, :] -= 2*np.array([eval_g(g['R'], xi[-1],  yj[j]) for j in range(Ny)]) / (alpha['R']*hx)
        if alpha['B'] != 0: r[:,  0] -= 2*np.array([eval_g(g['B'], xi[i],   yj[0]) for i in range(Nx)]) / (alpha['B']*hy)
        if alpha['T'] != 0: r[:, -1] -= 2*np.array([eval_g(g['T'], xi[i],   yj[-1])for i in range(Nx)]) / (alpha['T']*hy)
        return r.reshape(-1)
    if dim==3:
        Nx, Ny, Nz = N+2, M+2, L+2
        r = f_vals.copy().reshape(Nx, Ny, Nz).astype(float)
        YY, ZZ   = np.meshgrid(yj, zk, indexing='ij')
        XX, ZZ2  = np.meshgrid(xi, zk, indexing='ij')
        XX2, YY2 = np.meshgrid(xi, yj, indexing='ij')
        def F(f, A, B): return np.vectorize(f)(A, B) if callable(f) else np.full(A.shape, float(f))
        if alpha['L'] != 0: r[0,  :, :] -= 2*F(lambda y,z: eval_g(g['L'],xi[0],  y,z), YY, ZZ)  / (alpha['L']*hx)
        if alpha['R'] != 0: r[-1, :, :] -= 2*F(lambda y,z: eval_g(g['R'],xi[-1], y,z), YY, ZZ)  / (alpha['R']*hx)
        if alpha['B'] != 0: r[:,  0, :] -= 2*F(lambda x,z: eval_g(g['B'],x,yj[0], z), XX, ZZ2) / (alpha['B']*hy)
        if alpha['T'] != 0: r[:, -1, :] -= 2*F(lambda x,z: eval_g(g['T'],x,yj[-1],z), XX, ZZ2) / (alpha['T']*hy)
        if alpha['D'] != 0: r[:, :,  0] -= 2*F(lambda x,y: eval_g(g['D'],x,y,zk[0]),  XX2,YY2) / (alpha['D']*hz)
        if alpha['U'] != 0: r[:, :, -1] -= 2*F(lambda x,y: eval_g(g['U'],x,y,zk[-1]), XX2,YY2) / (alpha['U']*hz)
        return r.reshape(-1)


def dir_mask_val(dim, N, alpha, beta, g, xi, M=None, yj=None, L=None, zk=None):
    def dir(s): return alpha[s] == 0
    if dim == 1:
        Nx = N+2;  mask = np.zeros(Nx, dtype=bool);  uD = np.zeros(Nx)
        if dir('L'): mask[0]  = True;  uD[0]  = eval_g(g['L'], xi[0])  / beta['L']
        if dir('R'): mask[-1] = True;  uD[-1] = eval_g(g['R'], xi[-1]) / beta['R']
        return mask, uD
    if dim == 2:
        Nx, Ny = N+2, M+2
        mask = np.zeros((Nx, Ny), dtype=bool);  uD = np.zeros((Nx, Ny))
        if dir('L'): mask[0,  :] = True;  uD[0,  :] = [eval_g(g['L'], xi[0],   yj[j])/beta['L'] for j in range(Ny)]
        if dir('R'): mask[-1, :] = True;  uD[-1, :] = [eval_g(g['R'], xi[-1],  yj[j])/beta['R'] for j in range(Ny)]
        if dir('B'): mask[:,  0] = True;  uD[:,  0] = [eval_g(g['B'], xi[i],   yj[0]) /beta['B'] for i in range(Nx)]
        if dir('T'): mask[:, -1] = True;  uD[:, -1] = [eval_g(g['T'], xi[i],   yj[-1])/beta['T'] for i in range(Nx)]
        return mask.reshape(-1), uD.reshape(-1)
    if dim == 3:
        Nx, Ny, Nz = N+2, M+2, L+2
        mask = np.zeros((Nx, Ny, Nz), dtype=bool);  uD = np.zeros((Nx, Ny, Nz))
        YY, ZZ   = np.meshgrid(yj, zk, indexing='ij')
        XX, ZZ2  = np.meshgrid(xi, zk, indexing='ij')
        XX2, YY2 = np.meshgrid(xi, yj, indexing='ij')
        def F(f, A, B): return np.vectorize(f)(A, B) if callable(f) else np.full(A.shape, float(f))
        if dir('L'): mask[0,  :, :] = True;  uD[0,  :, :] = F(lambda y,z: eval_g(g['L'],xi[0],  y,z), YY, ZZ)  / beta['L']
        if dir('R'): mask[-1, :, :] = True;  uD[-1, :, :] = F(lambda y,z: eval_g(g['R'],xi[-1], y,z), YY, ZZ)  / beta['R']
        if dir('B'): mask[:,  0, :] = True;  uD[:,  0, :] = F(lambda x,z: eval_g(g['B'],x,yj[0], z), XX, ZZ2) / beta['B']
        if dir('T'): mask[:, -1, :] = True;  uD[:, -1, :] = F(lambda x,z: eval_g(g['T'],x,yj[-1],z), XX, ZZ2) / beta['T']
        if dir('D'): mask[:, :,  0] = True;  uD[:, :,  0] = F(lambda x,y: eval_g(g['D'],x,y,zk[0]),  XX2,YY2) / beta['D']
        if dir('U'): mask[:, :, -1] = True;  uD[:, :, -1] = F(lambda x,y: eval_g(g['U'],x,y,zk[-1]), XX2,YY2) / beta['U']
        return mask.reshape(-1), uD.reshape(-1)
        
def apply_dir(A, rhs, mask, uD):
    A = A.tolil()
    for i in np.where(mask)[0]:
        A.rows[i] = [i]
        A.data[i] = [1.0]
    rhs = rhs.copy()
    rhs[mask] = uD[mask]
    return A.tocsr(), rhs


def applyRobinConv1D(C, b, alpha, beta, id0, id1):
    if alpha[id0] != 0:
        R = beta[id0] / alpha[id0]
        C[0, :]  = 0
        C[0, 0]  =  b * R
    if alpha[id1] != 0:
        R = beta[id1] / alpha[id1]
        C[-1, :]  = 0
        C[-1, -1] = -b * R
    return C

def FDconv_homo_1D(N, h, b):
    C = diags([-b/(2*h), b/(2*h)], [-1, 1], shape=(N, N), format='lil')
    C[0, :] = 0;  C[-1, :] = 0
    return C.tocsr()

def FDconv_rob_1D(N, h, b, alpha, beta, id0, id1):
    return applyRobinConv1D(FDconv_homo_1D(N, h, b).tolil(), b, alpha, beta, id0, id1).tocsr()

def FDconv_sparse(dim, N, hx, bx, alpha, beta,
                  M=None, hy=None, by=None, L=None, hz=None, bz=None):
    if dim == 1:
        return FDconv_rob_1D(N+2, hx, bx, alpha, beta, 'L', 'R')
    if dim == 2:
        Nx, Ny = N+2, M+2
        Cx = FDconv_rob_1D(Nx, hx, bx, alpha, beta, 'L', 'R')
        Cy = FDconv_rob_1D(Ny, hy, by, alpha, beta, 'B', 'T')
        return kron(Cx, eye(Ny)) + kron(eye(Nx), Cy)
    if dim == 3:
        Nx, Ny, Nz = N+2, M+2, L+2
        Cx = FDconv_rob_1D(Nx, hx, bx, alpha, beta, 'L', 'R')
        Cy = FDconv_rob_1D(Ny, hy, by, alpha, beta, 'B', 'T')
        Cz = FDconv_rob_1D(Nz, hz, bz, alpha, beta, 'D', 'U')
        return (kron(kron(Cx, eye(Ny)), eye(Nz)) +
                kron(kron(eye(Nx), Cy), eye(Nz)) +
                kron(kron(eye(Nx), eye(Ny)), Cz))

def rhs_conv_robin(dim, N, bx, alpha, g, xi,
                   M=None, by=None, yj=None,
                   L=None, bz=None, zk=None):
    if dim == 1:
        r = np.zeros(N+2)
        if alpha['L'] != 0: r[0]  += bx * eval_g(g['L'], xi[0])  / alpha['L']
        if alpha['R'] != 0: r[-1] -= bx * eval_g(g['R'], xi[-1]) / alpha['R']
        return r
    if dim == 2:
        Nx, Ny = N+2, M+2
        r = np.zeros((Nx, Ny))
        if alpha['L'] != 0: r[0,  :] += bx * np.array([eval_g(g['L'], xi[0],  yj[j]) for j in range(Ny)]) / alpha['L']
        if alpha['R'] != 0: r[-1, :] -= bx * np.array([eval_g(g['R'], xi[-1], yj[j]) for j in range(Ny)]) / alpha['R']
        if alpha['B'] != 0: r[:,  0] += by * np.array([eval_g(g['B'], xi[i],  yj[0]) for i in range(Nx)]) / alpha['B']
        if alpha['T'] != 0: r[:, -1] -= by * np.array([eval_g(g['T'], xi[i],  yj[-1])for i in range(Nx)]) / alpha['T']
        return r.reshape(-1)
    if dim == 3:
        Nx, Ny, Nz = N+2, M+2, L+2
        r = np.zeros((Nx, Ny, Nz))
        YY, ZZ   = np.meshgrid(yj, zk, indexing='ij')
        XX, ZZ2  = np.meshgrid(xi, zk, indexing='ij')
        XX2, YY2 = np.meshgrid(xi, yj, indexing='ij')
        def F(f, A, B): return np.vectorize(f)(A, B) if callable(f) else np.full(A.shape, float(f))
        if alpha['L'] != 0: r[0,  :, :] += bx * F(lambda y,z: eval_g(g['L'],xi[0],  y,z), YY, ZZ)  / alpha['L']
        if alpha['R'] != 0: r[-1, :, :] -= bx * F(lambda y,z: eval_g(g['R'],xi[-1], y,z), YY, ZZ)  / alpha['R']
        if alpha['B'] != 0: r[:,  0, :] += by * F(lambda x,z: eval_g(g['B'],x,yj[0], z), XX, ZZ2) / alpha['B']
        if alpha['T'] != 0: r[:, -1, :] -= by * F(lambda x,z: eval_g(g['T'],x,yj[-1],z), XX, ZZ2) / alpha['T']
        if alpha['D'] != 0: r[:, :,  0] += bz * F(lambda x,y: eval_g(g['D'],x,y,zk[0]),  XX2,YY2) / alpha['D']
        if alpha['U'] != 0: r[:, :, -1] -= bz * F(lambda x,y: eval_g(g['U'],x,y,zk[-1]), XX2,YY2) / alpha['U']
        return r.reshape(-1)
    


def div_I_grad_u(u_f, I_f, M, N, hx, hy, active_mask):

    Nu, Mv = N + 2, M + 2
    nn=Nu*Mv
    u_full = np.zeros(nn); u_full[active_mask] = u_f
    I_full = np.zeros(nn); I_full[active_mask] = I_f

    u = u_full.reshape(Nu, Mv)
    I = I_full.reshape(Nu, Mv)

    # face-centered I
    Ixp = 0.5 * (I[1:Nu, :] + I[0:Nu-1, :])
    Ixm = Ixp.copy()
    Iyp = 0.5 * (I[:, 1:Mv] + I[:, 0:Mv-1])
    Iym = Iyp.copy()

    # grad u at faces
    du_dx_p = (u[1:Nu, :] - u[0:Nu-1, :]) / hx
    du_dx_m = du_dx_p.copy()
    du_dy_p = (u[:, 1:Mv] - u[:, 0:Mv-1]) / hy
    du_dy_m = du_dy_p.copy()

    # fluxes
    Fx_p = Ixp * du_dx_p
    Fx_m = Ixm * du_dx_m
    Fy_p = Iyp * du_dy_p
    Fy_m = Iym * du_dy_m

    # divergence on cell centers (1..Nu-2, 1..Mv-2)
    div = np.zeros_like(u)
    div[1:Nu-1, 1:Mv-1] = (
        (Fx_p[1:Nu-1, 1:Mv-1] - Fx_p[0:Nu-2, 1:Mv-1]) / hx +
        (Fy_p[1:Nu-1, 1:Mv-1] - Fy_p[1:Nu-1, 0:Mv-2]) / hy
    )

    return div.reshape(-1)[active_mask]


def apply_hole_robin_sparse(A, rhs, N, hx, M, hy,
                             i0, i1, j0, j1,
                             alpha_h, beta_h, g_h, xi, yj):
    Ny = M + 2
    rhs = rhs.copy()
    A   = A.tolil()

    def _al(s): return alpha_h[s] if isinstance(alpha_h, dict) else float(alpha_h)
    def _be(s): return beta_h[s]  if isinstance(beta_h,  dict) else float(beta_h)
    def _gv(s, x, y):
        gf = g_h[s] if isinstance(g_h, dict) else g_h
        return eval_g(gf, x, y)

    for i in range(i0, i1+1):
        for j in range(j0, j1+1):
            p = i*Ny + j
            A.rows[p] = [p];  A.data[p] = [1.0];  rhs[p] = 0.0

    al, be = _al('L'), _be('L')
    if al > 0:
        R = be/al
        for j in range(j0, j1+1):
            p = (i0-1)*Ny + j
            A[p, i0*Ny     + j] = 0
            A[p, (i0-2)*Ny + j] += 1/hx**2
            A[p, p]             -= 2*R/hx
            rhs[p] -= 2*_gv('L', xi[i0-1], yj[j]) / (al*hx)

    al, be = _al('R'), _be('R')
    if al > 0:
        R = be/al
        for j in range(j0, j1+1):
            p = (i1+1)*Ny + j
            A[p, i1*Ny     + j] = 0.0
            A[p, (i1+2)*Ny + j] += 1/hx**2
            A[p, p]             -= 2*R/hx
            rhs[p] -= 2*_gv('R', xi[i1+1], yj[j]) / (al*hx)

    al, be = _al('B'), _be('B')
    if al > 0:
        R = be/al
        for i in range(i0, i1+1):
            p = i*Ny + (j0-1)
            A[p, i*Ny + j0]     = 0.0
            A[p, i*Ny + (j0-2)] += 1/hy**2
            A[p, p]             -= 2*R/hy
            rhs[p] -= 2*_gv('B', xi[i], yj[j0-1]) / (al*hy)

    al, be = _al('T'), _be('T')
    if al > 0:
        R = be/al
        for i in range(i0, i1+1):
            p = i*Ny + (j1+1)
            A[p, i*Ny + j1]     = 0.0
            A[p, i*Ny + (j1+2)] += 1/hy**2
            A[p, p]             -= 2*R/hy
            rhs[p] -= 2*_gv('T', xi[i], yj[j1+1]) / (al*hy)

    return A.tocsr(), rhs


def hole_mask_2D(N, M, i0, i1, j0, j1):
    mask = np.zeros((N+2, M+2), dtype=bool)
    mask[i0:i1+1, j0:j1+1] = True
    return mask.reshape(-1)

def fill_hole_ghosts(vec, N, M, hx, hy, i0, i1, j0, j1, alpha_h, beta_h, g_h):
    V = vec.reshape(N+2, M+2).copy()

    # Left wall: ghost at col i0, boundary fluid at i0-1, mirror at i0-2
    al, be = alpha_h['L'], beta_h['L']
    V[i0, j0:j1+1] = V[i0-2, j0:j1+1] - 2*hx*(be/al)*V[i0-1, j0:j1+1] + 2*hx*g_h['L']/al

    # Right wall: ghost at col i1, boundary fluid at i1+1, mirror at i1+2
    al, be = alpha_h['R'], beta_h['R']
    V[i1, j0:j1+1] = V[i1+2, j0:j1+1] - 2*hx*(be/al)*V[i1+1, j0:j1+1] + 2*hx*g_h['R']/al

    # Bottom wall: ghost at row j0, boundary fluid at j0-1, mirror at j0-2
    al, be = alpha_h['B'], beta_h['B']
    V[i0:i1+1, j0] = V[i0:i1+1, j0-2] - 2*hy*(be/al)*V[i0:i1+1, j0-1] + 2*hy*g_h['B']/al

    # Top wall: ghost at row j1, boundary fluid at j1+1, mirror at j1+2
    al, be = alpha_h['T'], beta_h['T']
    V[i0:i1+1, j1] = V[i0:i1+1, j1+2] - 2*hy*(be/al)*V[i0:i1+1, j1+1] + 2*hy*g_h['T']/al

    return V.reshape(-1)



import math

def FDadv_1D(N, h, b):
    C = diags([-b/(2*h), b/(2*h)], [-1, 1], shape=(N, N), format='lil')
    return C.tocsr()

def solveHyp(tspan, x_span, u0, h, a, burgers, f, scheme, nu=0.9):
    k=h*nu/abs(a)

    x=np.arange(x_span[0]+h, x_span[1], h)

    N=len(x)
    M=math.ceil((tspan[1]-tspan[0])/k+1)    

    U0 = u0(x)

    B1 = FDadv_1D(N, h, 1)
    # Periodic BC
    B1[0, N-1] = -1/(2*h)
    B1[N-1, 0] = 1/(2*h)
    B=B1

    C1 = FDlapl_homo(1,N,h)
    # Periodic BC
    C1[0, N-1] = 1/(h**2)
    C1[N-1, 0] = 1/(h**2)
    C=C1

   
    if scheme == 'Upwind':  # Upwind
        if a<0:
            B = -B 
        coeff = a * h / 2 
    elif scheme == 'Lax-Friedrichs':  
        coeff = h**2 / (2*k)
    elif scheme == 'Lax-Wendroff':
        coeff = a**2 * k / 2

    # Time stepping
    U = np.zeros((N, M))
    U[:, 0] = U0

    for j in range(1, M):
        if scheme == 1 and a<0:  # Upwind with positive a
            U[:, j] = U[:, j-1] - (abs(a)*k * B.dot(U[:, j-1]) + (coeff*k) * C.dot(U[:, j-1]))
        elif scheme == 1 and a>0:  # Upwind with negative a
            if burgers:
                U[:, j] = U[:, j-1] - (a*k * B.dot(f(U[:, j-1])) - (coeff*k) * C.dot(U[:, j-1]))
            else:
                U[:, j] = U[:, j-1] - (a*k * B.dot(U[:, j-1]) - (coeff*k) * C.dot(U[:, j-1]))
        else:
            if burgers:
                U[:, j] = U[:, j-1] - (a*k * B.dot(f(U[:, j-1])) - (coeff*k) * C.dot(U[:, j-1]))
            else:
                U[:, j] = U[:, j-1] - (a*k * B.dot(U[:, j-1]) - (coeff*k) * C.dot(U[:, j-1]))

    return U


from custom_solvers import newton_solver, quasi_newton_solver
from scipy.optimize import fsolve

def MoL_CN_nwt(f, tspan, y0, N_steps, J=None, tau=1e-8, maxit=None):

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
                Jf = J(tk, x)
                from scipy.sparse.linalg import LinearOperator
                if isinstance(Jf, LinearOperator):
                    def mv(v):
                        return v - (h / 2.0) * Jf.matvec(v)
                    return LinearOperator((n, n), matvec=mv)
                Jf = np.asarray(Jf)
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

def MoL_CN_quasi(f, tspan, y0, N_steps, tau=1e-5, maxit=None):
    return MoL_CN_nwt(f, tspan, y0, N_steps, J=None, tau=tau, maxit=maxit)

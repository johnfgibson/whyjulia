from numpy import *
from numpy.fft import fft
from numpy.fft import ifft
from pylab import *
import timeit


def ksnorm(u) :
    """ksnorm(u) : sqrt(1/Lx int_0^Lx u^2 dx)"""
    N = size(u)
    s = 0
    for i in range(0, N) :
        s += u[i]*u[i]
    return sqrt(s/N)
    

def ksbenchmark(Nx, printnorm=False) :
    """ksbenchmark: benchmark the KS-CNAB2 algorithm for Nx gridpoints"""

    Lx = Nx/16*pi   # spatial domain [0, L] periodic
    dt = 1.0/16     # discrete time step 
    T  = 200.0      # integrate from t=0 to t=T
    Nt  = T/dt       # total number of time steps

    x = (Lx/Nx)*arange(0,Nx)
    u0 = cos(x) + 0.1*sin(x/8) + 0.01*cos((2*pi/Lx)*x);
    u = 0

    Nruns = 5
    skip = 1 
    avgtime = 0

    for n in range(Nruns) :
        tic = timeit.default_timer()
        u = ksintegrate(u0,Lx,dt, Nt)
        toc = timeit.default_timer()
        print ("cputime == ", toc - tic)
        avgtime += toc - tic

    avgtime /= Nruns

    if printnorm :
        print("norm(u(0)) == ", ksnorm(u0))
        print("norm(u(T)) == ", ksnorm(u))

    print("avgtime == ", avgtime)

def ksintegrate(u, Lx, dt, Nt) : 
    """ksintegrate: integrate kuramoto-sivashinsky equation (Python)
        u_t = -u*u_x - u_xx - u_xxxx, domain x in [0,Lx], periodic BCs 

    inputs
          u = initial condition (vector of u(x) values on uniform gridpoints))
         Lx = domain length
         dt = time step
         Nt = number of integration timesteps
      nsave = save every nsave-th time step
 
    outputs
          u = final state, vector of u(x, Nt*dt) at uniform x gridpoints
    """
   
    Nx = size(u);
    kx = concatenate((arange(0,Nx/2), array([0]), arange(-Nx/2+1,0))) # int wavenumbers: exp(2*pi*kx*x/L)
    alpha = 2*pi*kx/Lx;               # real wavenumbers:    exp(alpha*x)
    D = 1j*alpha;                     # D = d/dx operator in Fourier space
    L = pow(alpha,2) - pow(alpha,4);  # linear operator -D^2 - D^3 in Fourier space
    G = -0.5*D;                       # -1/2 D operator in Fourier space

    # Express PDE as u_t = Lu + N(u), L is linear part, N nonlinear part.
    # Then Crank-Nicolson Adams-Bashforth discretization is 
    # 
    # (I - dt/2 L) u^{n+1} = (I + dt/2 L) u^n + 3dt/2 N^n - dt/2 N^{n-1}
    #
    # let A = (I - dt/2 L) 
    #     B = (I + dt/2 L), then the CNAB timestep formula 
    # 
    # u^{n+1} = A^{-1} (B u^n + 3dt/2 N^n - dt/2 N^{n-1}) 

    # some convenience variables
    dt2  = dt/2;
    dt32 = 3*dt/2;
    A =      ones(Nx) + dt2*L;
    B = 1.0/(ones(Nx) - dt2*L)

    Nn  = G*fft(u*u); # compute -u u_x (spectral), notation Nn  = N^n     = N(u(n dt))
    Nn1 = Nn;         #                            notation Nn1 = N^{n-1} = N(u((n-1) dt))
    u = fft(u);       # transform u (spectral)

    # timestepping loop
    for n in range(0,int(Nt)) :

        Nn1 = Nn;                        # shift nonlinear term in time: N^{n-1} <- N^n
        uu = real(ifft(u))
        uu = uu*uu
        uu = fft(uu)
        Nn  = G*uu    # compute Nn == -u u_x (spectral)

        u = B * (A * u + dt32*Nn - dt2*Nn1);

    return real(ifft(u));


#if __name__ == "__main__":
#    # execute only if run as a script
#    print sys.argv[1]
#    #print sys.argv[2]
#
#    ksbenchmark(int(sys.argv[1]))


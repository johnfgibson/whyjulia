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
        Nn  = G*fft(real(ifft(u*u)));    # compute Nn == -u u_x (spectral)

        u = B * (A * u + dt32*Nn - dt2*Nn1);

    return real(ifft(u));

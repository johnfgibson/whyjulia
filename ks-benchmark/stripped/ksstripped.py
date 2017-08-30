def ksintegrate(u, Lx, dt, Nt) : 
    Nx = size(u)
    kx = concatenate((arange(0,Nx/2), array([0]), arange(-Nx/2+1,0)))
    alpha = 2*pi*kx/Lx
    D = 1j*alpha
    L = pow(alpha,2) - pow(alpha,4)
    G = -0.5*D;
    dt2  = dt/2
    dt32 = 3*dt/2
    A =      ones(Nx) + dt2*L
    B = 1.0/(ones(Nx) - dt2*L)
    Nn  = G*fft(u*u)
    Nn1 = Nnn
    u = fft(u)
    for n in range(0,int(Nt)+1) :
        Nn1 = Nn
        Nn  = G*fft(real(ifft(u*u)))
        u = B * (A * u + dt32*Nn - dt2*Nn1)
    return real(ifft(u))

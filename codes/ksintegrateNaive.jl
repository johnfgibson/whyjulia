"""
ksintegrateNaive: integrate kuramoto-sivashinsky equation (Julia)
       u_t = -u*u_x - u_xx - u_xxxx, domain x in [0,Lx], periodic BCs 

 inputs
          u = initial condition (vector of u(x) values on uniform gridpoints))
         Lx = domain length
         dt = time step
         Nt = number of integration timesteps
      nsave = save every nsave-th time step

 outputs
          u = final state, vector of u(x, Nt*dt) at uniform x gridpoints

This a line-by-line translation of a Matlab code into Julia. It uses out-of-place
FFTs and doesn't pay any attention to the allocation of temporary vectors within
the time-stepping loop. Hence the name "naive".
"""
function ksintegrateNaive(u, Lx, dt, Nt)
    Nx = length(u)                  # number of gridpoints
    x = collect(0:(Nx-1)/Nx)*Lx
    kx = vcat(0:Nx/2-1, 0, -Nx/2+1:-1)  # integer wavenumbers: exp(2*pi*kx*x/L)
    alpha = 2*pi*kx/Lx              # real wavenumbers:    exp(alpha*x)
    D = 1im*alpha                   # D = d/dx operator in Fourier space
    L = alpha.^2 - alpha.^4         # linear operator -D^2 - D^4 in Fourier space
    G = -0.5*D                      # -1/2 D operator in Fourier space

    # Express PDE as u_t = Lu + N(u), L is linear part, N nonlinear part.
    # Then Crank-Nicolson Adams-Bashforth discretization is 
    # 
    # (I - dt/2 L) u^{n+1} = (I + dt/2 L) u^n + 3dt/2 N^n - dt/2 N^{n-1}
    #
    # let A = (I - dt/2 L) 
    #     B = (I + dt/2 L), then the CNAB timestep formula 
    # 
    # u^{n+1} = A^{-1} (B u^n + 3dt/2 N^n - dt/2 N^{n-1}) 

    # convenience variables
    dt2  = dt/2
    dt32 = 3*dt/2
    A =  ones(Nx) + dt2*L
    B = (ones(Nx) - dt2*L).^(-1)

    Nn  = G.*fft(u.*u) # -u u_x (spectral), notation Nn = N^n     = N(u(n dt))
    Nn1 = copy(Nn)     #                   notation Nn1 = N^{n-1} = N(u((n-1) dt))
    u  = fft(u)        # transform u to spectral

    # timestepping loop
    for n = 0:Nt
        Nn1 = copy(Nn)                 # shift nonlinear term in time: N^{n-1} <- N^n
        Nn  = G.*fft(real(ifft(u)).^2) # compute Nn = -u u_x

        u = B .* (A .* u + dt32*Nn - dt2*Nn1)
    end

    real(ifft(u))
end


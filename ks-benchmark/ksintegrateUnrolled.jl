"""
ksintegrateUnrolled: integrate kuramoto-sivashinsky equation (Julia)
  

      u_t = -u*u_x - u_xx - u_xxxx, domain x in [0,Lx], periodic BCs 

 inputs
          u = initial condition (vector of u(x) values on uniform gridpoints))
         Lx = domain length
         dt = time step
         Nt = number of integration timesteps
      nsave = save every nsave-th time step

 outputs
          u = final state, vector of u(x, Nt*dt) at uniform x gridpoints

This implementation has one improvements over ksintegrateInplace.jl. It explicitly
unrolls all the vector equations into for loops. For some reason this works
better (runs faster) than the unrolled loops produced by the Julia compiler. 
"""
function ksintegrateUnrolled(u, Lx, dt, Nt)
    u = (1+0im)*u                       # force u to be complex
    Nx = length(u)                      # number of gridpoints
    kx = vcat(0:Nx/2-1, 0:0, -Nx/2+1:-1)# integer wavenumbers: exp(2*pi*kx*x/L)
    alpha = 2*pi*kx/Lx                  # real wavenumbers:    exp(alpha*x)
    D = 1im*alpha                       # spectral D = d/dx operator 
    L = alpha.^2 - alpha.^4             # spectral L = -D^2 - D^4 operator
    G = -0.5*D                          # spectral -1/2 D operator, to eval -u u_x = 1/2 d/dx u^2

    # convenience variables
    dt2  = dt/2
    dt32 = 3*dt/2
    A =  ones(Nx) + dt2*L
    B = (ones(Nx) - dt2*L).^(-1)

    # compute FFTW plans
    FFT! = plan_fft!(u, flags=FFTW.ESTIMATE)
    IFFT! = plan_ifft!(u, flags=FFTW.ESTIMATE)

    # compute nonlinear term Nn == N(u^n) = -u u_x and Nn1 
    Nn  = G.*fft(u.^2) # Nn == -1/2 d/dx (u^2) = -u u_x, spectral
    Nn1 = copy(Nn)     # Nn1 == N(u^{n-1}). For first timestep, let Nn1 = Nn
    FFT!*u

    # timestepping loop
    for n = 0:Nt

        Nn1 .= Nn
        Nn .= u

        IFFT!*Nn # in-place FFT

        @inbounds for i = 1:length(Nn)
            @fastmath Nn[i] = Nn[i]*Nn[i]
        end

        FFT!*Nn

        @inbounds for i = 1:length(Nn)
            @fastmath Nn[i] = G[i]*Nn[i]
        end

        @inbounds for i = 1:length(u)
            @fastmath u[i] = B[i]* (A[i] * u[i] + dt32*Nn[i] - dt2*Nn1[i])
        end

    end

    IFFT!*u
    real(u)
end

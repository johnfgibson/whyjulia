using FFTW

"""
  ksbenchmark(Nx, ksintegrator): benchmark a kuramoto-sivashinksy integration algorithm
    on Nx gridpoints. Usage example: ksbenchmark(512, ksintegrateNaive)
"""
function ksbenchmark(Nx, ksintegrator, Nruns, printnorms=false)

    Lx = Nx/16*pi             # spatial domain [0, L] periodic
    dt = 1/16                 # discrete time step 
    T  = 200                  # integrate from t=0 to t=T
    nplot = round(Int,1/dt)   # save every nploth time step
    Nt = round(Int, T/dt)     # total number of timesteps
 
    x = Lx*(0:Nx-1)/Nx
    u0 = cos.(x) + 0.1*sin.(x/8) + 0.01*cos.((2*pi/Lx)*x) 
    u = copy(u0)

    skip = 1
    avgtime = 0
    
    for run=1:Nruns
        tic()
        u .= ksintegrator(u0, Lx, dt, Nt)
        cputime = toc()
        if run > skip
            avgtime = avgtime + cputime
        end
    end
    
    
    if printnorms
        @show ksnorm(u0)
        @show ksnorm(u)
    end

    avgtime = avgtime/(Nruns-skip)
    @show avgtime

end

function ksnorm(u)
    s = 0.0
    for n = 1:length(u)
        s += (u[n])^2
    end
    sqrt(s/length(u))
end

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
    D = 1im*alpha                  # D = d/dx operator in Fourier space
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
    for n = 1:Nt
        Nn1 = copy(Nn)                 # shift nonlinear term in time: N^{n-1} <- N^n
        Nn  = G.*fft(real(ifft(u)).^2) # compute Nn = -u u_x

        u = B .* (A .* u + dt32*Nn - dt2*Nn1)
    end

    real(ifft(u))
end

"""
ksintegrate: integrate kuramoto-sivashinsky equation (Julia)
       u_t = -u*u_x - u_xx - u_xxxx, domain x in [0,Lx], periodic BCs 

 inputs
          u = initial condition (vector of u(x) values on uniform gridpoints))
         Lx = domain length
         dt = time step
         Nt = number of integration timesteps
      nsave = save every nsave-th time step

 outputs

          u = final state, vector of u(x, Nt*dt) at uniform x gridpoints

This implementation has two improvements over ksintegrateNaive.jl. It uses 
   (1) in-place FFTs
   (2) loop fusion: Julia can translate arithmetic vector expressions in dot syntax 
       to single for loop over the components, which should be much faster than
       constructing a temporary vector for each operation in the vector expression. 
"""
function ksintegrateInplace(u, Lx, dt, Nt)
    u = (1+0im)*u                       # force u to be complex
    Nx = length(u)                      # number of gridpoints
    kx = vcat(0:Nx/2-1, 0:0, -Nx/2+1:-1)# integer wave #s,  exp(2*pi*i*kx*x/L)
    alpha = 2*pi*kx/Lx                  # real wavenumbers, exp(i*alpha*x)
    D = 1im*alpha                       # spectral D = d/dx operator 
    L = alpha.^2 - alpha.^4             # spectral L = -D^2 - D^4 operator
    G = -0.5*D                          # spectral -1/2 D operator, to eval -u u_x = 1/2 d/dx u^2

    # convenience variables
    dt2  = dt/2
    dt32 = 3*dt/2
    A_inv = (ones(Nx) - dt2*L).^(-1)
    B     =  ones(Nx) + dt2*L

    # compute FFTW plans
    FFT! = plan_fft!(u, flags=FFTW.ESTIMATE)
    IFFT! = plan_ifft!(u, flags=FFTW.ESTIMATE)

    # compute nonlinear term Nu == -u u_x and Nuprev (Nu at prev timestep)
    Nu     = G.*fft(u.^2) # Nu == -1/2 d/dx (u^2) = -u u_x
    Nuprev = copy(Nu)      # use Nuprev = Nu at first time step
    FFT!*u

    # timestepping loop
    for n = 1:Nt

        Nuprev .= Nu   # shift nonlinear term in time
        Nu .= u         # put u into N in prep for comp of nonlineat
        
        IFFT!*Nu       # transform Nu to gridpt values, in place
        Nu .= Nu.*Nu   # collocation calculation of u^2
        FFT!*Nu        # transform Nu back to spectral coeffs, in place

        Nu .= G.*Nu

        # loop fusion! Julia translates the folling line of code to a single for loop. 
        u .= A_inv .*(B .* u .+ dt32.*Nu .- dt2.*Nuprev) 
    end

    IFFT!*u
    real(u)
end

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

    # compute nonlinear term Nu == -u u_x and Nuprev (Nu at prev timestep)
    Nn  = G.*fft(u.^2) # Nnf == -1/2 d/dx (u^2) = -u u_x, spectral
    Nn1 = copy(Nn)     # use Nnf1 = Nnf at first time step
    FFT!*u

    # timestepping loop
    for n = 1:Nt

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

""" 
   Construct initial conditions for benchmarking ksintegrate* algorithms
   Useful for using Julia's benchmark utilities. 
"""
function ksinitconds(Nx)
    Lx = Nx/16*pi             # spatial domain [0, L] periodic
    dt = 1/16                 # discrete time step 
    T  = 200                  # integrate from t=0 to t=T
    nplot = round(Int,1/dt)   # save every nploth time step
    Nt = round(Int, T/dt)     # total number of timesteps
 
    x = Lx*(0:Nx-1)/Nx
    u0 = cos.(x) + 0.1*sin.(x/8) + 0.01*cos.(x/16) 

    u0, Lx, dt, Nt
end

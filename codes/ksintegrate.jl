function ksintegrateInplace(u, Lx, dt, Nt, nsave);
    u = (1+0im)*u                       # force u to be complex
    Nx = length(u)                      # number of gridpoints
    kx = vcat(0:Nx/2-1, 0:0, -Nx/2+1:-1)# integer wavenumbers: exp(2*pi*kx*x/L)
    alpha = 2*pi*kx/Lx                  # real wavenumbers:    exp(alpha*x)
    D = 1im*alpha                       # spectral D = d/dx operator 
    L = alpha.^2 - alpha.^4             # spectral L = -D^2 - D^4 operator
    G = -0.5*D                          # spectral -1/2 D operator

    Nsave = div(Nt, nsave)+1            # number of saved time steps
    t = (0:Nsave)*(dt*nsave)            # t timesteps
    U = zeros(Nsave, Nx)                # matrix of u(xⱼ, tᵢ) values
    U[1,:] = u                          # assign initial condition to U
    s = 2                               # counter for saved data
 
    # some convenience variables
    dt2  = dt/2
    dt32 = 3*dt/2
    A_inv = (ones(Nx) - dt2*L).^(-1)
    B     =  ones(Nx) + dt2*L
    
    # compute in-place FFTW plans
    FFT! = plan_fft!(u, flags=FFTW.ESTIMATE)
    IFFT! = plan_ifft!(u, flags=FFTW.ESTIMATE)

    # compute nonlinear term Nn = -u u_x 
    Nn  = G.*fft(u.^2);    # Nn = -1/2 d/dx u^2 = -u u_x
    Nn1 = copy(Nn);        # Nn1 = Nn at first time step
    FFT!*u;                # transform u physical -> spectral
    
    # timestepping loop
    for n = 1:Nt

        Nn1 .= Nn       # shift nonlinear term in time

        Nn .= u         # put u into Nn 
        IFFT!*Nn;       # transform Nn = u to gridpt values, in place
        Nn .= Nn.*Nn;   # collocation calculation, set Nn = u^2
        FFT!*Nn;        # transform Nn = u^2 back to spectral coeffs
        Nn .= G.*Nn;    # apply G = -1/2d/dx  to compute Nn = -1/2 d/dx (u^2)

        # loop fusion! Julia translates this line into a single for-loop on
        #   u[i] = A_inv[i] * (B[i]*u[i] + dt32*Nn[i] - dt2*Nn1[i]; 
        # no temporary vectors! 

        u .= A_inv .* (B .* u .+ dt32.*Nn .- dt2.*Nn1); 
        
        if mod(n, nsave) == 0
            U[s,:] = real(ifft(u))
            s += 1            
        end
    end
   
    t,U
end

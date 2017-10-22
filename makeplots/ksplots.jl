using Plots
gr()

function makeksplot()
    # set parameters
    Lx = 64*pi
    Nx = 1024
    dt = 1/16
    nsave = 8
    Nt = 3200

    # set initial condition and run simulation
    x = Lx*(0:Nx-1)/Nx
    u = cos.(x) + 0.1*sin.(x/8) + 0.01*cos.((2*pi/Lx)*x)
    t,U = ksintegrate(u, Lx, dt, Nt, nsave);

    # plot results
    Plots.heatmap(x,t,U, xlim=(x[1], x[end]), ylim=(t[1], t[end]), xlabel="x", ylabel="t", title="u(x,t)", fillcolor=:bluesreds)
end

function makescalingplot()
    d = readdlm("makeplots/cputime.asc")
    Nx = d[:,1]
    Plots.plot( Nx, d[:,7], label="Python", marker=:circ, color="green")
    Plots.plot!(Nx, d[:,5], label="Matlab", marker=:circ, color="red")
#    Plots.plot!(Nx, d[:,9], label="Julia naive", marker=:circ, color="orange")
    Plots.plot!(Nx, d[:,10], label="Julia", marker=:circ, color="orange")
    Plots.plot!(Nx, d[:,11], label="Julia unrolled", marker=:circ, color="yellow")
    Plots.plot!(Nx, d[:,3], label="C", marker=:circ, color="lightblue")
    Plots.plot!(Nx, d[:,4], label="C++", marker=:circ, color="blue")
    Plots.plot!(Nx, d[:,2], label="Fortran", linestyle=:solid, color="black")

    N  = [2.0^6, 2.0^17];
    Plots.plot!(N, 1e-05*N .* log10.(N), label="Nx log Nx", xlabel="Nx", ylabel="cpu time", linestyle=:dash, color="black")
    Plots.plot!(yscale=:log10, xscale=:log10,xlim=(10,3e05), ylim=(1e-03,1e02))
    #Plots.plot!(title="KS-CNAB2 benchmarks")
    Plots.plot!(legend=:topleft)
end

function makelinecountplot()
    d = readdlm("makeplots/linecount.asc")
    Nx = d[:,1]
    c = 1/d[8,1]
    Plots.plot([d[1,2]], [c*d[1,1]],  label="Python", marker=:circ, color="green")
    Plots.plot!([d[2,2]], [c*d[2,1]], label="Matlab", marker=:circ, color="red" )
#    Plots.plot!([d[3,2]], [c*d[3,1]], label="Julia naive", marker=:circ, color="red")
    Plots.plot!([d[4,2]], [c*d[4,1]], label="Julia", marker=:circ, color="orange")
    Plots.plot!([d[5,2]], [c*d[5,1]], label="Julia unrolled", marker=:circ, color="yellow")
    Plots.plot!([d[6,2]], [c*d[6,1]], label="Fortran", marker=:circ, color="black")
    Plots.plot!([d[7,2]], [c*d[7,1]], label="C++", marker=:circ, color="blue")
    Plots.plot!([d[8,2]], [c*d[8,1]], label="C", marker=:circ, color="lightblue")
    Plots.plot!(xlabel="lines of code", ylabel="cpu time, C = 1", xlim=(0,80), ylim=(0,3))
end


function ksintegrate(u0, Lx, dt, Nt, nsave);
    u = (1+0im)*u0                      # force u to be complex
    Nx = length(u)                      # number of gridpoints
    kx = vcat(0:Nx/2-1, 0:0, -Nx/2+1:-1)# integer wavenumbers: exp(2*pi*kx*x/L)
    alpha = 2*pi*kx/Lx                  # real wavenumbers:    exp(alpha*x)
    D = 1im*alpha                       # spectral D = d/dx operator 
    L = alpha.^2 - alpha.^4             # spectral L = -D^2 - D^4 operator
    G = -0.5*D                          # spectral -1/2 D operator
    
    Nsave = div(Nt, nsave)+1            # number of saved time steps, including t=0
    t = (0:Nsave)*(dt*nsave)            # t timesteps
    U = zeros(Nsave, Nx)                # matrix of u(xⱼ, tᵢ) values
    U[1,:] = u                          # assign initial condition to U
    s = 2                               # counter for saved data
    
    # convenience variables
    dt2  = dt/2
    dt32 = 3*dt/2
    A_inv = (ones(Nx) - dt2*L).^(-1)
    B     =  ones(Nx) + dt2*L
    
    # compute in-place FFTW plans
    FFT! = plan_fft!(u, flags=FFTW.ESTIMATE)
    IFFT! = plan_ifft!(u, flags=FFTW.ESTIMATE)

    # compute nonlinear term Nn == -u u_x 
    Nn  = G.*fft(u.^2);    # Nn == -1/2 d/dx (u^2) = -u u_x
    Nn1 = copy(Nn);        # Nn1 = Nn at first time step
    FFT!*u;
    
    # timestepping loop
    for n = 1:Nt

        Nn1 .= Nn       # shift nonlinear term in time
        Nn .= u         # put u into Nn in prep for comp of nonlinear term
        
        IFFT!*Nn;       # transform Nn to gridpt values, in place
        Nn .= Nn.*Nn;   # collocation calculation of u^2
        FFT!*Nn;        # transform Nn back to spectral coeffs, in place

        Nn .= G.*Nn;    # compute Nn = N(u) = -1/2 d/dx u^2 = -u u_x

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



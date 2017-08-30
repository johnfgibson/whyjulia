function ksintegrateInplace(u0, Lx, dt, Nt)
    u = (1+0im)*u0
    Nx = length(u)
    kx = vcat(0:Nx/2-1, 0:0, -Nx/2+1:-1)
    alpha = 2*pi*kx/Lx
    D = 1im*alpha
    L = alpha.^2 - alpha.^4
    G = -0.5*D
    dt2  = dt/2
    dt32 = 3*dt/2
    A_inv = (ones(Nx) - dt2*L).^(-1)
    B     =  ones(Nx) + dt2*L
    FFT! = plan_fft!(u, flags=FFTW.ESTIMATE)
    IFFT! = plan_ifft!(u, flags=FFTW.ESTIMATE)
    Nn  = G.*fft(u.^2)
    Nn1 = copy(Nn)
    FFT!*u
    for n = 0:Nt
        Nn1 .= Nn
        Nn .= u
        IFFT!*Nn
        Nn .= Nn.*Nn
        FFT!*Nn
        Nn .= G.*Nn
        u .= A_inv .*(B .* u .+ dt32.*Nn .- dt2.*Nn1) 
    end
    IFFT!*u
    real(u)
end

function ksintegrateUnrolled(u, Lx, dt, Nt)
    u = (1+0im)*u
    Nx = length(u)
    kx = vcat(0:Nx/2-1, 0:0, -Nx/2+1:-1)
    alpha = 2*pi*kx/Lx
    D = 1im*alph
    L = alpha.^2 - alpha.^4
    G = -0.5*D
    dt2  = dt/2
    dt32 = 3*dt/2
    A =  ones(Nx) + dt2*L
    B = (ones(Nx) - dt2*L).^(-1)
    FFT! = plan_fft!(u, flags=FFTW.ESTIMATE)
    IFFT! = plan_ifft!(u, flags=FFTW.ESTIMATE)
    Nn  = G.*fft(u.^2)
    Nn1 = copy(Nn)
    FFT!*u
    for n = 0:Nt
        copy!(Nn1, Nn)
        copy!(Nn,  u)
        IFFT!*Nn
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

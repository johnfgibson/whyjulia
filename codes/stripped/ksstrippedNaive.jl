function ksintegrateNaive(u, Lx, dt, Nt)
    Nx = length(u)
    kx = vcat(0:Nx/2-1, 0, -Nx/2+1:-1)
    alpha = 2*pi*kx/Lx
    D = 1im*alpha
    L = alpha.^2 - alpha.^4
    G = -0.5*D
    dt2  = dt/2
    dt32 = 3*dt/2
    A =  ones(Nx) + dt2*L
    B = (ones(Nx) - dt2*L).^(-1)
    Nn  = G.*fft(u.*u)
    Nn1 = Nn
    u  = fft(u)
    for n = 0:Nt
        Nn1 = Nn
        Nn  = G.*fft(real(ifft(u)).^2)
        u = B .* (A .* u + dt32*Nn - dt2*Nn1)
    end
    real(ifft(u))
end

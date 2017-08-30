function u = ksstripped(u, Lx, dt, Nt, nsave)
  Nx = length(u);
  kx = [0:Nx/2-1 0 -Nx/2+1:-1];
  alpha = 2*pi*kx/Lx;
  D = i*alpha;
  L = alpha.^2 - alpha.^4;
  G = -0.5*D;
  dt2  = dt/2;
  dt32 = 3*dt/2;
  A_inv = (ones(1,Nx) - dt2*L).^(-1);
  B     =  ones(1,Nx) + dt2*L;
  Nn  = G.*fft(u.*u);
  Nn1 = Nn;
  u = fft(u);
  for n = 1:Nt
    Nn1 = Nn;
    Nn  = G.*fft(real(ifft(u)).^2);
    u = A_inv .* (B .* u + dt32*Nn - dt2*Nn1);
  end
  u = real(ifft(u))
end

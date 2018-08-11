function t,U = ksintegrate(u, Lx, dt, Nt, nsave)
  Nx = length(u);                % number of gridpoints
  kx = [0:Nx/2-1 0 -Nx/2+1:-1];  % integer wavenumbers: exp(2*pi*kx*x/L)
  alpha = 2*pi*kx/Lx;            % real wavenumbers:    exp(alpha*x)
  D = i*alpha;                   % D = d/dx operator in Fourier space
  L = alpha.^2 - alpha.^4;       % linear operator -D^2 - D^3 in Fourier space
  G = -0.5*D;                    % -1/2 D operator in Fourier space

  Nsave = round(Nt/nsave) + 1    % number of saved time steps
  t = (0:Nsave)*(dt*nsave)       % t timesteps
  U = zeros(Nsave, Nx)           % matrix of u(xⱼ, tᵢ) values
  U(1,:) = u                     % assign initial condition to U
  s = 2                          % counter for saved data
   
  % some convenience variables
  dt2  = dt/2;
  dt32 = 3*dt/2;
  A_inv = (ones(1,Nx) - dt2*L).^(-1);
  B     =  ones(1,Nx) + dt2*L;

  % compute nonlinear term Nn = -u u_x
  Nn  = G.*fft(u.*u);    % Nn = -1/2 d/dx u^2 = -u u_x
  Nn1 = Nn;              % Nn1 = Nn at first time step
  u = fft(u);            % transform u physical -> spectral
  
  % timestepping loop  
  for n = 1:Nt
      
    Nn1 = Nn;                       % shift nonlinear term in time
    Nn  = G.*fft(real(ifft(u)).^2); % compute Nn = N(u) = -u u_x

    % time-stepping eqn: Matlab generates six temp vectors to evaluate
    u = A_inv .* (B .* u + dt32*Nn - dt2*Nn1);

    if mod(n, nsave) == 0
       U(s,:) = real(ifft(u));
          s = s+1;            
    end

end

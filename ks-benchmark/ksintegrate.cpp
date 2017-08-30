typedef std::complex<double> Complex;

// ksintegrate: integrate kuramoto-sivashinsky equation (Python)
//       u_t = -u*u_x - u_xx - u_xxxx, domain x in [0,Lx], periodic BCs 

//  inputs
//        u = initial condition (vector of u(x) values on uniform gridpoints))
//       Lx = domain length
//       dt = time step
//       Nt = number of integration timesteps
//    nsave = save every nsave-th time step
// 
//  outputs
//        u = final state, vector of u(x, Nt*dt) at uniform x gridpoints
   
void ksintegrate(Complex* u, const int Nx, const double Lx, const double dt, const int Nt) {

  int* kx        = new int[Nx];     // for Nx=8, kx = [0 1 2 3 0 -3 -2 -1]
  double*  alpha = new double[Nx];  // real wavenumbers:    exp(alpha*x)
  Complex* D     = new Complex[Nx]; // D = d/dx operator in Fourier space
  Complex* G     = new Complex[Nx]; // -1/2 D operator in Fourier space
  double*  L     = new double[Nx];    // -D^2 - D^4

  // Assign Fourier wave numbers in order that FFTW produces
  for (int n=0; n<Nx/2; ++n)      
    kx[n] = n;

  kx[Nx/2] = 0.0;                 

  for (int n=Nx/2+1; n<Nx; ++n)   
    kx[n] = -Nx + n;


  // Assign alpha, D, G, L
  const double a = 2*M_PI/Lx;
  for (int n=0; n<Nx; ++n) {   
    alpha[n] = a*kx[n];   
    double alpha2 = alpha[n]*alpha[n];  // alpha[n]^2
    L[n] = alpha2*(1-alpha2);
    G[n] = Complex(0.0, -0.5*alpha[n]);
  }

  // CNAB formula
  // (I - dt/2 L) u^{n+1} = (I + dt/2 L) u^n + 3dt/2 N^n - dt/2 N^{n-1}
  //
  // u^{n+1} = (I - dt/2 L)^{-1} [(I + dt/2 L) u^n + 3dt/2 N^n - dt/2 N^{n-1}]
  //
  // let A = (I + dt/2 L)
  // let B = (I - dt/2 L)^{-1}

  // some convenience variables
  double dt2  = dt/2;
  double dt32 = 3*dt/2;
  double Nx_inv = 1.0/Nx;
  double Nx_inv2 = Nx_inv*Nx_inv;

  double* A = new double[Nx];
  double* B = new double[Nx];

  for (int n=0; n<Nx; ++n) {
    A[n] = 1.0 + dt2*L[n];
    B[n] = 1.0/(1.0 - dt2*L[n]);
  }

  Complex* uu = new Complex[Nx];  
  for (int n=0; n<Nx; ++n) 
    uu[n] = u[n]*u[n];

  // Cast Native C++ complex array pointers to FFTW equivalent.
  fftw_complex* u_fftw  =  reinterpret_cast<fftw_complex*>(u);
  fftw_complex* uu_fftw =  reinterpret_cast<fftw_complex*>(uu);
  
  // Construct FFTW plans
  fftw_plan u_fftw_plan   = fftw_plan_dft_1d(Nx, u_fftw,  u_fftw,  FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_plan u_ifftw_plan  = fftw_plan_dft_1d(Nx, u_fftw,  u_fftw,  FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_plan uu_fftw_plan  = fftw_plan_dft_1d(Nx, uu_fftw, uu_fftw, FFTW_FORWARD,  FFTW_ESTIMATE);
  fftw_plan uu_ifftw_plan = fftw_plan_dft_1d(Nx, uu_fftw, uu_fftw, FFTW_BACKWARD, FFTW_ESTIMATE);

  // Compute  Nn = G*fft(u*u), (-u u_x, spectral)
  fftw_execute(uu_fftw_plan);   // uu physical -> spectral

  Complex* Nn = new Complex[Nx];  
  for (int n=0; n<Nx; ++n)      // compute N(u) = -1/2 d/dx u^2, spectral
    Nn[n] = G[n]*uu[n];

  Complex* Nn1 = new Complex[Nx];  

  fftw_execute(u_fftw_plan);    // transform u physical -> spectral

  // timestepping loop
  for (int s=0; s<Nt; ++s) {

    for (int n=0; n<Nx; ++n) {
      Nn1[n] = Nn[n];           // shift nonlinear term in time: N^{n-1} <- N^n
      uu[n] = u[n];             // copy u to uu (spectral)
    }      

    fftw_execute(uu_ifftw_plan); // uu (holding u) spectral -> physical
    for (int n=0; n<Nx; ++n) 
      uu[n] *= Nx_inv;           // normalize

    for (int n=0; n<Nx; ++n) 
      uu[n] = uu[n]*uu[n];       // compute uu = u^2 physical,

    fftw_execute(uu_fftw_plan);  // transform uu physical -> spectral

    for (int n=0; n<Nx; ++n) 
      Nn[n] = G[n]*uu[n];        // compute Nn == -u u_x, spectral
    
    for (int n=0; n<Nx; ++n) 
      u[n] = B[n] * (A[n] * u[n] + dt32*Nn[n] - dt2*Nn1[n]);

  }

  fftw_execute(u_ifftw_plan);    // transform uu physical -> spectral
  for (int n=0; n<Nx; ++n) 
    u[n] *= Nx_inv;              // normalize  

  fftw_destroy_plan(u_fftw_plan);
  fftw_destroy_plan(uu_fftw_plan);
  fftw_destroy_plan(u_ifftw_plan);
  fftw_destroy_plan(uu_ifftw_plan);

  delete[] kx;
  delete[] alpha;
  delete[] G;
  delete[] L;
  delete[] A;
  delete[] B;
  delete[] uu;
  delete[] Nn;
  delete[] Nn1;
}

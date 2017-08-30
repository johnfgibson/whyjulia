void ksintegrate(Complex* u, const int Nx, const double Lx, const double dt, const int Nt) {
  int* kx        = new int[Nx];
  double*  alpha = new double[Nx];
  Complex* D     = new Complex[Nx];
  Complex* G     = new Complex[Nx];
  double*  L     = new double[Nx];
  for (int n=0; n<Nx/2; ++n)      
    kx[n] = n;
  kx[Nx/2] = 0.0;                 
  for (int n=Nx/2+1; n<Nx; ++n)   
    kx[n] = -Nx + n;
  const double a = 2*M_PI/Lx;
  for (int n=0; n<Nx; ++n) {   
    alpha[n] = a*kx[n];   
    double alpha2 = alpha[n]*alpha[n];
    L[n] = alpha2*(1-alpha2);
    G[n] = Complex(0.0, -0.5*alpha[n]);
  }
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
  fftw_complex* u_fftw  =  reinterpret_cast<fftw_complex*>(u);
  fftw_complex* uu_fftw =  reinterpret_cast<fftw_complex*>(uu);
  fftw_plan u_fftw_plan   = fftw_plan_dft_1d(Nx, u_fftw,  u_fftw,  FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_plan u_ifftw_plan  = fftw_plan_dft_1d(Nx, u_fftw,  u_fftw,  FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_plan uu_fftw_plan  = fftw_plan_dft_1d(Nx, uu_fftw, uu_fftw, FFTW_FORWARD,  FFTW_ESTIMATE);
  fftw_plan uu_ifftw_plan = fftw_plan_dft_1d(Nx, uu_fftw, uu_fftw, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(uu_fftw_plan);
  Complex* Nn = new Complex[Nx];  
  for (int n=0; n<Nx; ++n)
    Nn[n] = G[n]*uu[n];
  Complex* Nn1 = new Complex[Nx];  
  fftw_execute(u_fftw_plan);
  for (int s=0; s<Nt; ++s) {
    for (int n=0; n<Nx; ++n) {
      Nn1[n] = Nn[n];
      uu[n] = u[n];
    }      
    fftw_execute(uu_ifftw_plan);
    for (int n=0; n<Nx; ++n) 
      uu[n] *= Nx_inv;
    for (int n=0; n<Nx; ++n) 
      uu[n] = uu[n]*uu[n];
    fftw_execute(uu_fftw_plan);
    for (int n=0; n<Nx; ++n) 
      Nn[n] = G[n]*uu[n];
    for (int n=0; n<Nx; ++n) 
      u[n] = B[n] * (A[n] * u[n] + dt32*Nn[n] - dt2*Nn1[n]);
  }
  fftw_execute(u_ifftw_plan);
  for (int n=0; n<Nx; ++n) 
    u[n] *= Nx_inv;
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

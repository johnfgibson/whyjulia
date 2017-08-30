#include <math.h>
#include <time.h>
#include <complex>
#include <fftw3.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

using namespace std;

typedef std::complex<double> Complex;

void ksintegrate(Complex* u, int Nx, double Lx, double dt, int Nt);
double ksnorm(Complex* u, int Nx);

int main(int argc, char* argv[]) {

  // to be set as command-line args
  int Nx = 0;
  int Nruns = 5;
  bool printnorms = false;

  if (argc < 2) {
    cout << "please provide one integer argument Nx\n [two optional args: Nruns printnorm]" << endl;
    exit(1);
  }
  Nx = atoi(argv[1]);  

  if (argc >= 3)
    Nruns = atoi(argv[2]);

  if (argc >= 4)
    printnorms = bool(atoi(argv[3]));

  cout << "Nx == " << Nx << endl;

  double dt = 1.0/16.0;
  double T  = 200.0;
  double Lx = (M_PI/16.0)*Nx;
  double dx = Lx/Nx;
  int Nt = (int)(T/dt);

  double* x = new double[Nx];
  Complex* u0 = new Complex[Nx];
  Complex* u  = new Complex[Nx];

  for (int n=0; n<Nx; ++n) {
    x[n] = n*dx;
    u0[n] = cos(x[n]) + 0.1*sin(x[n]/8) + 0.01*cos((2*M_PI/Lx)*x[n]);
  }
  //cout << "norm(u0) == " << ksnorm(u0,Nx) << endl;

  //ofstream xs("x.asc");
  //for (int n=0; n<Nx; ++n)
  //xs << x[n] << '\n';

  int skip = 1;
  double avgtime = 0.0;

  for (int run=0; run<Nruns; ++run) {

    for (int n=0; n<Nx; ++n)
      u[n] = u0[n];

    clock_t tic = clock();
    ksintegrate(u, Nx, Lx, dt, Nt);
    clock_t toc = clock();

    double cputime =((double)(toc - tic))/CLOCKS_PER_SEC;
    if (run >= skip) 
      avgtime += cputime;

    printf("cputtime == %f\n", cputime);
  }
  printf("norm(u(0)) == %f\n", ksnorm(u0, Nx)); 
  printf("norm(u(T)) == %f\n", ksnorm(u, Nx)); 

  
  avgtime /= (Nruns-skip);
  printf("avgtime == %f\n", avgtime);

  delete[] u0;
  delete[] u;
  delete[] x;
}

void ksintegrate(Complex* u, int Nx, double Lx, double dt, int Nt) {
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
    double alpha2 = alpha[n]*alpha[n];  // alpha[n]^2
    D[n] = Complex(0.0, alpha[n]);                     
    L[n] = alpha2*(1-alpha2);
    G[n] = Complex(0.0, -0.5*alpha[n]);
  }
  double dt2  = dt/2;
  double dt32 = 3*dt/2;
  double Nx_inv = 1.0/Nx;
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
      uu[n] = uu[n]*uu[n];       // compute uu = u^2 physical
    fftw_execute(uu_fftw_plan);  // transform uu physical -> spectral
    for (int n=0; n<Nx; ++n) 
      Nn[n] = G[n]*uu[n];        // compute Nn == -u u_x, spectral
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
  delete[] D;
  delete[] G;
  delete[] L;
  delete[] A;
  delete[] B;
  delete[] uu;
  delete[] Nn;
  delete[] Nn1;
}

inline double square(double x) {return x*x;}

double ksnorm(Complex* u, int Nx) {
  double s = 0.0;
  for (int n=0; n<Nx; ++n)
    s += square(abs(u[n]));
  return sqrt(s/Nx);
}

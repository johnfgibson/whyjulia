#include <math.h>
#include <time.h>
#include <complex.h>
#include <fftw3.h>
#include <stdlib.h>
#include <stdio.h>
//#include <stdbool.h>

typedef complex double Complex;

void ksintegrate(Complex* u, int Nx, double Lx, double dt, int Nt);
double ksnorm(Complex* u, int Nx);

const double pi = 3.14159265358979323846; /* why doesn't M_PI from math.j work? :-( */

int main(int argc, char* argv[]) {

  /* to be set as command-line args */
  int Nx = 0;         /* number of x gridpoints */
  int Nruns = 5;      /* number of benchmarking runs */
  int printnorms = 0; /* can't figure out C bool type :-( */

  if (argc < 2) {
    printf("please provide one integer argument Nx\n");
    exit(1);
  }
  Nx = atoi(argv[1]);  

  if (argc >= 3)
    Nruns = atoi(argv[2]);

  if (argc >= 4)
    printnorms = atoi(argv[3]);

  printf("Nx == %d\n", Nx);

  double dt = 1.0/16.0;
  double T  = 200.0;
  double Lx = (pi/16.0)*Nx;
  double dx = Lx/Nx;
  int Nt = (int)(T/dt);

  double* x   = malloc(Nx*sizeof(double));
  Complex* u0 = malloc(Nx*sizeof(Complex));
  Complex* u  = malloc(Nx*sizeof(Complex));

  for (int n=0; n<Nx; ++n) {
    x[n] = n*dx;
    u0[n] = cos(x[n]) + 0.1*sin(x[n]/8) + 0.01*cos((2*pi/Lx)*x[n]);
  }

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

  free(u0);
  free(u);
  free(x);
}

void ksintegrate(Complex* u, int Nx, double Lx, double dt, int Nt) {
  int* kx        = malloc(Nx*sizeof(int));
  double*  alpha = malloc(Nx*sizeof(double));  
  Complex* D     = malloc(Nx*sizeof(Complex)); 
  Complex* G     = malloc(Nx*sizeof(Complex)); 
  double*  L     = malloc(Nx*sizeof(double));  

  for (int n=0; n<Nx/2; ++n)      
    kx[n] = n;
  kx[Nx/2] = 0.0;                 
  for (int n=Nx/2+1; n<Nx; ++n)   
    kx[n] = -Nx + n;
  
  const double a = 2*pi/Lx;
  for (int n=0; n<Nx; ++n) {   
    alpha[n] = a*kx[n];   
    double alpha2 = alpha[n]*alpha[n];  /* alpha[n]^2 */
    D[n] = alpha[n]*I;                     
    L[n] = alpha2*(1-alpha2);
    G[n] = -0.5*alpha[n]*I;
  }
  double dt2  = dt/2;
  double dt32 = 3*dt/2;
  double Nx_inv = 1.0/Nx;
  double* A = malloc(Nx*sizeof(double));
  double* B = malloc(Nx*sizeof(double));
  for (int n=0; n<Nx; ++n) {
    A[n] = 1.0 + dt2*L[n];
    B[n] = 1.0/(1.0 - dt2*L[n]);
  }
  Complex* uu = malloc(Nx*sizeof(Complex));  
  for (int n=0; n<Nx; ++n) 
    uu[n] = u[n]*u[n];

  fftw_complex* u_fftw  =  u;
  fftw_complex* uu_fftw =  uu;
  fftw_plan u_fftw_plan   = fftw_plan_dft_1d(Nx, u_fftw,  u_fftw,  FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_plan u_ifftw_plan  = fftw_plan_dft_1d(Nx, u_fftw,  u_fftw,  FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_plan uu_fftw_plan  = fftw_plan_dft_1d(Nx, uu_fftw, uu_fftw, FFTW_FORWARD,  FFTW_ESTIMATE);
  fftw_plan uu_ifftw_plan = fftw_plan_dft_1d(Nx, uu_fftw, uu_fftw, FFTW_BACKWARD, FFTW_ESTIMATE);

  fftw_execute(uu_fftw_plan); 
  Complex* Nn = malloc(Nx*sizeof(Complex));
  for (int n=0; n<Nx; ++n) 
    Nn[n] = G[n]*uu[n];
  Complex* Nn1 = malloc(Nx*sizeof(Complex));

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
      uu[n] = uu[n]*uu[n];       /* compute uu = u^2 physical */
    fftw_execute(uu_fftw_plan);  /* transform uu physical -> spectral */
    for (int n=0; n<Nx; ++n) 
      Nn[n] = G[n]*uu[n];        /* compute Nn == -u u_x, spectral */
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
  free(kx);
  free(alpha);
  free(D);
  free(G);
  free(L);
  free(A);
  free(B);
  free(uu);
  free(Nn);
  free(Nn1);
}

inline double square(double x) {return x*x;}

double ksnorm(Complex* u, int Nx) {
  double s = 0.0;
  for (int n=0; n<Nx; ++n)
    s += square(cabs(u[n]));
  return sqrt(s/Nx);
}

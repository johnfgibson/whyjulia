 module KS
   implicit none
   save
   integer,          parameter :: Nx = 512
   double precision, parameter :: dt = 1d0/16d0
   double precision, parameter :: T  = 200d0
   double precision, parameter :: pi = 3.14159265358979323846d0
   double precision, parameter :: Lx = (pi/16d0)*Nx
 end module KS
 subroutine ksintegrate(Nt, u)
   use KS
   implicit none
   integer,          intent(in)    :: Nt
   double complex,   intent(inout) :: u(0:Nx-1)
   double precision :: dt2, dt32, Nx_inv
   double precision :: A(0:Nx-1), B(0:Nx-1), L(0:Nx-1), kx(0:Nx-1)
   double complex   :: alpha(0:Nx-1), G(0:Nx-1), Nn(0:Nx-1), Nn1(0:Nx-1)
   double complex, save :: u_(0:Nx-1), uu(0:Nx-1) 
   double complex, save :: us(0:Nx-1), uus(0:Nx-1)
   logical, save :: planned=.false.
   integer*8 :: plan_u2us, plan_us2u, plan_uu2uus, plan_uus2uu
   integer :: n, fftw_patient=32, fftw_estimate=64, fftw_forward=-1, fftw_backward=1
   if(.not.planned) then
      call dfftw_plan_dft_1d(plan_u2us, Nx, u_, us, fftw_forward, fftw_estimate)
      call dfftw_plan_dft_1d(plan_us2u, Nx, us, u_, fftw_backward, fftw_estimate)
      call dfftw_plan_dft_1d(plan_uu2uus, Nx, uu, uus, fftw_forward, fftw_estimate)
      call dfftw_plan_dft_1d(plan_uus2uu, Nx, uus, uu, fftw_backward, fftw_estimate)
      planned = .true.
   end if
   do n = 0, Nx/2-1
      kx(n) = n
   end do
   kx(Nx/2) = 0d0
   do n = Nx/2+1, Nx-1
      kx(n) = -Nx + n;
   end do
   alpha = (2d0*pi/Lx)*kx
   L = alpha**2 - alpha**4
   G = -0.5d0*dcmplx(0d0,1d0)*alpha
   Nx_inv = 1d0/Nx
   dt2  = dt/2d0
   dt32 = 3d0*dt/2d0
   A = 1d0 + dt2*L
   B = 1d0/(1d0 - dt2*L)
   uu = u*u
   call dfftw_execute(plan_uu2uus)
   Nn  = G*uus
   Nn1 = Nn
   u_ = u
   call dfftw_execute(plan_u2us)
   do n = 1, Nt
      Nn1 = Nn
      call dfftw_execute(plan_us2u)
      u_ = u_ * Nx_inv
      uu = u_*u_
      call dfftw_execute(plan_uu2uus)
      Nn = G*uus
      us = B * (A*us + dt32*Nn - dt2*Nn1)
   end do
   call dfftw_execute(plan_us2u)
   u = u_ * Nx_inv
 end subroutine ksintegrate

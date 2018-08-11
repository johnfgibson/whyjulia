# Why Julia?

Julia is the future of scientific computing.
This talk is my effort to show why, and to convince people it's worth the time and effort to switch from other languages such as Matlab, Python, C, C++, and Fortran.
In particular, I think we should stop teaching those languages to undergraduate scientists and engineers, and start teaching them Julia.
This talk is aimed at people with experience in linear algebra, ordinary and partial differential equations. Comparisons are made to the languages mentioned above. 

## Abstract

Julia is an innovative new open-source programming language for high-level, high-performance numerical computing.
Julia combines the ease-of-use, numeric focus, and graphics of Matlab, the speed of C and Fortran,
the general-purpose breadth and extensibility of Python, and the metaprogramming power of Lisp.
Julia uses type inference and just-in-time compilation to compile high-level user code to machine code on the fly.
A rich set of numeric types and extensive numerical libraries are built-in.
As a result, Julia is competitive with Matlab for interactive exploration and with C and Fortran for high-performance computing.
This talk is largely live demos of Julia's innovative features, plus a benchmark of Julia against C, C++, Fortran, Matlab, and Python on a spectral time-stepping algorithm for a 1d nonlinear partial differential equation.
The Julia PDE code is nearly as compact as Matlab and nearly as fast as Fortran.

## The talk: [Julia, the future of scientific computing](1-whyjulia.ipynb)

## Outline

  1. Julia: easy as Matlab
     * linear algebra (Ax=b)
     * ordinary differential equations (Lorenz)
  2. Julia: fast as Fortran
     * partial differential equations (Kuramoto-Sivanshinksy)
     * performance and line count comparison (Python, Matlab, Julia, C, C++, Fortran)
  3. Julia: easy, dynamic, and fast. How?
     * just-in-time compilation (iterated logistic map)
     * type inference
  4. Julia: flexible as Python
     * comprehensive numeric types
     * high-precision math (singular values of Hilbert matrix)
     * user-defined types (linear algebra over Galois field)
  5 Julia: deep as Lisp
     * expressions: code as data
     * macros: code that transforms code
  6. Parallelism
     * remote procedure calls
     * parallel loops
     * shared arrays
     * distributed arrays

## Acknowledgments

This talk owes a lot to
  * David Sanders' [Hands on Julia](https://github.com/dpsanders/hands_on_julia)
  * Chris Rackauckas' [Intro to Julia](https://github.com/UCIDataScienceInitiative/IntroToJulia)
  * Andreas Noack's [Fast and Flexible Linear Algebra in Julia](https://www.youtube.com/watch?v=VS0fnUOAKpI) 

   
   

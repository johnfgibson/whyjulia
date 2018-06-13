#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <ctime>

using namespace std;

inline double f(double x) {
  return 4*x*(1-x);
}

int main(int argc, char* argv[]) {
  double x = argc > 1 ? atof(argv[1]) : 0.0;

  double t0 = clock();
  for (int n=0; n<1000000; ++n)
    x = f(x);
  double t1 = clock();

  cout << "t = " << (t1-t0)/CLOCKS_PER_SEC << " seconds" << endl;
  cout << setprecision(17);
  cout << "x = " << x << endl;
  
  return 0;
}
  

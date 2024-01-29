#include <iostream>
#include <string>

#include "PlanarRmc.hpp"
// #include "rand.hpp"

using namespace std;

int main() {
  int n, Lx, Ly, n_iter_max;
  float a, cutoff, r_par, thresh;
  string filename;
  cout << "number of clusters: 40" << endl;
  n = 40;  // cin >> n;
  cout << "x width of model space: 50" << endl;
  Lx = 50;  // cin >> Lx;
  cout << "y width of model space: 50" << endl;
  Ly = 50;  // cin >> Ly;
  cout << "lattice constant: 0.32" << endl;
  a = 0.32;  // cin >> a;
  cout << "cutoff distance: 0.65" << endl;
  cutoff = 0.65;  // cin >> cutoff;
  cout << "particle radius: 0.32" << endl;
  r_par = 0.32;  // cin >> r_par;
  cout << "max iteration: 100" << endl;
  n_iter_max = 100;  // cin >> n_iter_max;
  cout << "filename: ../../../data/s191/s191sPNc/s191PNc95.csv" << endl;
  filename = "../../../data/s191/s191sPNc/s191PNc95.csv";  // cin >> filename;

  PlanarRmc::TrianglePlanarSimulator sim;
  sim.init(n, Lx, Ly, a, cutoff, r_par);
  cout << "init done" << endl;
  sim.load_exp_data(filename);
  cout << "load_exp_data done" << endl;
  cout << "initial mse: " << sim.mse() << endl;
  thresh = sim.mse() * 0.1;
  sim.run(n_iter_max, thresh);
  cout << "run done" << endl;
  cout << "final mse: " << sim.mse() << endl;
  sim.save("test");
  return 0;
}
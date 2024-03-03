
#include <chrono>
#include <iostream>
#include <string>

#include "rmc.cpp"

using namespace std;

int main() {
  chrono::system_clock::time_point start, now;
  start = chrono::system_clock::now();
  Simulator sim;
  string src, dst;
  int Lx, Ly, n, max_iter, move_per_step;
  double res_thresh, sigma2;
  double q_min = 4.0, q_max = 7.0;

  cout << "Enter the source file name : ";
  cin >> src;
  sim.load_exp_data(src);
  sim.set_q_range(q_min, q_max);
  now = chrono::system_clock::now();
  cout << "Data loaded in "
       << chrono::duration_cast<chrono::milliseconds>(now - start).count()
       << "ms" << endl;

  cout << "Enter the destination file name : ";
  cin >> dst;

  cout << "Lx : ";
  cin >> Lx;
  cout << "Ly : ";
  cin >> Ly;
  cout << "n : ";
  cin >> n;
  // Lx = 90;
  // Ly = 90;
  // n = 465;
  // sim.set_Lx(Lx);
  // sim.set_Ly(Ly);
  // sim.set_n(n);
  // sim.init();
  // test text
  string srcxtl = "test3.xtl";
  sim.load_xtl(srcxtl);

  cout << "max_iter : ";
  cin >> max_iter;
  cout << "res_thresh : ";
  cin >> res_thresh;
  cout << "sigma2 coeff: ";
  cin >> sigma2;

  now = chrono::system_clock::now();
  cout << "Initialized in "
       << chrono::duration_cast<chrono::milliseconds>(now - start).count()
       << "ms" << endl;

  res_thresh *= sim.get_residual();
  sigma2 = sim.get_residual() * sigma2;
  move_per_step = 3;

  sim.run(max_iter, res_thresh, sigma2, move_per_step);
  now = chrono::system_clock::now();
  cout << max_iter << " steps finished in "
       << chrono::duration_cast<chrono::milliseconds>(now - start).count()
       << "ms" << endl;

  sim.save_result(dst);
  now = chrono::system_clock::now();
  cout << "Result saved in "
       << chrono::duration_cast<chrono::milliseconds>(now - start).count()
       << "ms" << endl;
  return 0;
}

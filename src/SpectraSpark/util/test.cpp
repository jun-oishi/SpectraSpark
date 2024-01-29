#include <iostream>

#include "util.hpp"

using namespace util;
using namespace std;

void test_mse() {
  cout << "test mse" << endl;
  Eigen::VectorXf x(3);
  x << 1, 2, 3;
  Eigen::VectorXf y(3);
  y << 1, 2, 3;
  cout << mse(x, y) << "(expected 0.000000)" << endl;
  y << 1, 2, 4;
  cout << mse(x, y) << "(expected 0.333333)" << endl;
  y << 2, 5, 1;
  cout << mse(x, y) << "(expected 4.666666)" << endl;

  x.resize(1000);
  y.resize(1000);
  x.setRandom();
  y = x * 0;
  cout << mse(x, y) << "(expected around 0.333333)" << endl;
}

int main() {
  test_mse();
  return 0;
}
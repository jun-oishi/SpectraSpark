#include "PlanarRmc.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include "../util/util.hpp"
#include "rand.hpp"

#define DEBUG

#ifdef DEBUG
#define print(mes) \
  if (__debug) std::cout << mes << std::endl;

#define showvar(var) \
  if (__debug) std::cout << #var << " = " << var << std::endl;
#endif

namespace PlanarRmc {
const int N_THETA = 180;
const float THETA_STEP = M_PI / N_THETA;

using namespace Eigen;
using namespace RmcCore;

using lu = long unsigned int;
bool __debug = false;

void TrianglePlanarModel::init(int n, int Lx, int Ly, float a, float cutoff) {
  this->n = n;
  this->Lx = Lx * a, this->Ly = Ly * a * sqrt(3) / 2;
  this->a = a;
  this->Lxi = Lx, this->Lyi = Ly;
  this->cutoff = cutoff;
  this->coord.resize(n);

  float _a_particle = cutoff * cutoff * M_PI;
  float _whole_area = Lx * Ly * a * a * sqrt(3) / 2;
  if (n > Lx * Ly || cutoff > max((double)Lx * a / 2, Ly * a * sqrt(3) / 2) ||
      n * _a_particle > _whole_area * 0.8) {
    throw runtime_error("abort initialization for too busy clusters");
  }
  cout << "_max_rand_iter:" << _max_rand_iter << endl;

  coord.at(0) = {randint() % Lx, randint() % Ly};
  for (int i = 1; i < n; i++) {
    _n_iter = 0;
    coord.at(i) = {randint() % Lx, randint() % Ly};
    for (int j = 0; j < i; j++) {
      if (is_in_cutoff(i, j)) {
        i--;
        _n_iter++;
        if (_n_iter > _max_rand_iter) {
          throw runtime_error("Fail to initialize");
        }
        break;
      }
    }
  }
  return;
}

void TrianglePlanarModel::move() {
  int i = randint() % n;
  Vector2i before = coord.at(i);
  Vector2i step;
  int dir = randint() % 6;
  switch (dir) {
    case 0:
      step = {1, 0};
      break;
    case 1:
      step = {0, 1};
      break;
    case 2:
      step = {-1, 1};
      break;
    case 3:
      step = {-1, 0};
      break;
    case 4:
      step = {0, -1};
      break;
    case 5:
      step = {1, -1};
      break;
  }
  coord.at(i) += step;
  if (coord.at(i)(0) >= Lxi) {
    coord.at(i)(0) -= Lxi;
  } else if (coord.at(i)(0) < 0) {
    coord.at(i)(0) += Lxi;
  }
  if (coord.at(i)(1) >= Lyi) {
    coord.at(i)(1) -= Lyi;
  } else if (coord.at(i)(1) < 0) {
    coord.at(i)(1) += Lyi;
  }
  i_last_moved = i;
  last_move = coord.at(i) - before;
  showvar(i);
  showvar(last_move);
  return;
}

void TrianglePlanarModel::undo() {
  coord.at(i_last_moved) -= last_move;
  return;
}

Vector2f TrianglePlanarModel::real_coord(int i) const {
  Vector2i c = coord.at(i);
  return Vector2f(c(0) * this->a, c(1) * this->a * sqrt(3) / 2);
}

float TrianglePlanarModel::dist2(int i, int j) const {
  Vector2f dr = real_coord(i) - real_coord(j);
  if (dr(0) > Lx / 2) {
    dr(0) -= Lx;
  } else if (dr(0) < -Lx / 2) {
    dr(0) += Lx;
  }

  if (dr(1) > Ly / 2) {
    dr(1) -= Ly;
  } else if (dr(1) < -Ly / 2) {
    dr(1) += Ly;
  }

  return dr.squaredNorm();
}

bool TrianglePlanarModel::is_in_cutoff(int i, int j) const {
  return dist2(i, j) < cutoff * cutoff;
}

void TrianglePlanarModel::load(const string &filename) {
  vector<vector<int>> data;
  util::loadtxt(filename, data, "#", ',');
  this->n = data.size();
  coord.resize(n);
  for (int i = 0; i < n; i++) {
    coord.at(i) = {data[i][0], data[i][1]};
  }
  return;
}

void TrianglePlanarModel::save(const string &filename) const {
  string header = "x,y";
  vector<vector<int>> data(n, vector<int>(2));
  for (int i = 0; i < n; i++) {
    data[i][0] = real_coord(i)(0);
    data[i][1] = real_coord(i)(1);
  }
  util::savetxt(filename, data, header, "#", ',');
}

void TrianglePlanarSimulator::init(int n, int Lx, int Ly, float a, float cutoff,
                                   float r_par) {
  model.init(n, Lx, Ly, a, cutoff);
  this->r_par = r_par;
  return;
}

void TrianglePlanarSimulator::load_exp_data(const string &filename) {
  bool __debug = true;
  exp_data.load(filename);
  sim_data.setQ(exp_data.q());
  sim_data.setI(Eigen::VectorXf(exp_data.size()));
  sim_data_before.setQ(exp_data.q());
  sim_data_before.setI(Eigen::VectorXf(exp_data.size()));
  print("data loaded");

  this->__re_exp_qr.resize(exp_data.size());
  this->__im_exp_qr.resize(exp_data.size());
  for (lu i_q = 0; i_q < exp_data.size(); i_q++) {
    this->__re_exp_qr[i_q].resize(N_THETA, this->n());
    this->__im_exp_qr[i_q].resize(N_THETA, this->n());
  }
  for (int i = 0; i < this->n(); i++) {
    this->__compute_exp_qr(i);
  }

  this->__compute_i_par();
  print("__compute_i_par done");

  this->compute_i();
  print("compute_i done")

      this->mse_history.resize(1);
  print("mse_history resized") this->mse_history[0] =
      util::mse(exp_data.i(), sim_data.i());
  showvar(this->mse());
  return;
}

void TrianglePlanarSimulator::compute_i() {
  this->sim_data_before.setI(sim_data.i());
  Eigen::VectorXf I(exp_data.size());

  for (lu i_q = 0; i_q < exp_data.size(); i_q++) {
    float i = this->__i_par[i_q];
    for (int i_theta; i_theta < N_THETA; i_theta++) {
      float theta = i_theta * THETA_STEP;
      float re = 0, im = 0;
      for (int i = 0; i < this->n(); i++) {
        re += this->__re_exp_qr[i_q](i_theta, i);
        im += this->__im_exp_qr[i_q](i_theta, i);
      }
      I(i_q) += i * (re * re + im * im) * sin(theta) * THETA_STEP;
    }
  }
  sim_data.setI(I);
  return;
}

void TrianglePlanarSimulator::__compute_exp_qr(int i) {
  for (lu i_q = 0; i_q < exp_data.size(); i_q++) {
    for (int i_theta = 0; i_theta < N_THETA; i_theta++) {
      double theta = i_theta * THETA_STEP;
      Vector2f q = exp_data.q()(i_q) * Vector2f(cos(theta), sin(theta));
      double qr = q.dot(model.real_coord(i));
      this->__re_exp_qr[i_q](i_theta, i) = cos(qr);
      this->__im_exp_qr[i_q](i_theta, i) = sin(qr);
    }
  }
}

void TrianglePlanarSimulator::__compute_i_par() {
  this->__i_par.resize(exp_data.size());
  // ref.Matsuoka, Nihon Kessho Gakkaishi 1999
  for (lu i_q = 0; i_q < exp_data.size(); i_q++) {
    double x = exp_data.q()(i_q) * r_par;
    double tmp = 3 * (sin(x) - x * cos(x)) / (x * x * x);
    this->__i_par[i_q] = tmp * tmp;
  }
}

void TrianglePlanarSimulator::move() {
  bool __debug = true;
  model.move();
  int i = model.get_i_last_moved();
  this->__compute_exp_qr(i);
  this->compute_i();
  double new_mse = util::mse(exp_data.i(), sim_data.i());
  print("mse:" << mse() << " -> " << new_mse);

  // ref. McGreevy 2001
  if (new_mse < mse() || rand_uniform() < exp((-new_mse + mse()) / 2)) {
    print("accepted");
    this->mse_history.push_back(new_mse);
  } else {
    print("rejected");
    model.undo();
    sim_data.setI(sim_data_before.i());
    this->mse_history.push_back(mse());
  }
  return;
}

int TrianglePlanarSimulator::run(int n_step_max, double thresh) {
  for (int i = 0; i < n_step_max; i++) {
    this->move();
    if (mse() < thresh) return 0;
  }
  return 1;
}

void TrianglePlanarSimulator::save(const string &filename) const {
  model.save(filename + "_cl_coord.csv");
  sim_data.save(filename + "_sim.csv");

  string header = "step,mse";
  vector<vector<double>> data(mse_history.size());
  for (lu i = 0; i < mse_history.size(); i++) {
    data[i] = {mse_history[i]};
  }
  util::savetxt(filename + "_mse_history.csv", data, header, "#", ',', 1);

  return;
}

}  // namespace PlanarRmc
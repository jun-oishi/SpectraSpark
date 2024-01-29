#include <random>

namespace RmcCore {

const int __MAX_RANDINT__ = 2147483647;

std::mt19937 __engine;

void set_seed(int seed) {
  __engine.seed(seed);
  return;
};

int randint() {
  static std::uniform_int_distribution<> dist(0, __MAX_RANDINT__);
  return dist(__engine);
};

float rand_uniform() { return randint() / __MAX_RANDINT__; };

float rand_norm() {
  static std::normal_distribution<> dist(-1, 1);
  return dist(__engine);
};

}  // namespace RmcCore

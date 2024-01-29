#ifndef __RMC_RAND_HPP__
#define __RMC_RAND_HPP__

namespace RmcCore {

extern const int __MAX_RANDINT__;

/**
 * @brief Set the seed object
 *
 * @param seed
 */
void set_seed(int seed);

/**
 * @brief 一様乱数の整数を生成する
 *
 * @return int
 */
int randint();

/**
 * @brief [0,1]の一様乱数を生成する
 *
 * @return float
 */
float rand_uniform();

/**
 * @brief 平均0, 分散1の正規分布に従う乱数を生成する
 *
 * @return float
 */
float rand_norm();

}  // namespace RmcCore

#endif  // __RMC_RAND_HPP__
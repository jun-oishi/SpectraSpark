
#ifndef UTIL_HPP
#define UTIL_HPP

// #define EIGEN_NO_DEBUG
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace util {

/**
 * @brief csvファイルを読み込む
 *
 * @param filename
 * @param data
 * @param comments
 * @param delimiter
 */
template <typename T>
void loadtxt(const std::string &filename, std::vector<std::vector<T>> &data,
             const std::string comments = "#", const char delimiter = ' ');

template <typename T>
/**
 * @brief csvファイルに書き込む
 *
 * @param filename
 * @param data
 * @param header
 * @param comments
 * @param delimiter
 * @param idx_column
 */
void savetxt(const std::string &filename,
             const std::vector<std::vector<T>> &data,
             const std::string header = "", const std::string comments = "#",
             const char delimiter = ' ', const int idx_column = 0);

/**
 * @brief 二乗平均誤差を計算する
 *
 * @param x
 * @param y
 * @return double
 */
double mse(const Eigen::VectorXf &x, const Eigen::VectorXf &y);

}  // namespace util

#endif  // UTIL_HPP
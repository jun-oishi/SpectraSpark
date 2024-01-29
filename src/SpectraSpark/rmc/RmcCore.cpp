
#include "RmcCore.hpp"

#include <fstream>
#include <sstream>

#include "../util/util.hpp"

namespace RmcCore {

using lu = long unsigned int;

template <typename X>
X Model<X>::get_coord(int i) {
  return coord.at(i);
}

/**
 * @brief SAXSの実験データを読み込む
 * "#"から始まるヘッダ行を無視してcsvとして読み込み1,2列目をq, Iに格納する
 *
 * @param filename
 */
void SaxsData::load(const string &filename) {
  vector<vector<float>> data;
  util::loadtxt(filename, data, "#", ',');
  this->Q = Eigen::VectorXf(data.size());
  this->I = Eigen::VectorXf(data.size());
  for (lu i = 0; i < data.size(); i++) {
    this->Q(i) = data[i][0];
    this->I(i) = data[i][1];
  }
  this->I /= this->I.maxCoeff();
  return;
}

/**
 * @brief SAXSのデータを保存する
 * q, Iを1,2列目に持つcsvファイルを作成する
 *
 * @param filename
 */
void SaxsData::save(const string &filename) const {
  vector<vector<float>> data(this->size(), vector<float>(2));
  for (lu i = 0; i < this->size(); i++) {
    data[i][0] = this->Q(i);
    data[i][1] = this->I(i);
  }
  string header = "q[nm^-1],I[nm^-1]";
  util::savetxt(filename, data, header, "#", ',');
  return;
}

}  // namespace RmcCore
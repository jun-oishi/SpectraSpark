#ifndef RMC_CORE_HPP
#define RMC_CORE_HPP

// #define EIGEN_NO_DEBUG
#include <Eigen/Dense>
#include <exception>
#include <vector>

namespace RmcCore {

using namespace std;

template <typename X>
/**
 * @brief 粒子の座標を扱うモデルの基底クラス
 */
class Model {
 public:
  virtual void move() = 0;  // ランダムに粒子を一つ選んで動かす
  X get_coord(int i);       // i番目の粒子の座標を取得
  int _max_rand_iter = 1000000;  // 乱数の再生成回数の上限
  /**
   * @brief 粒子数を取得する
   *
   * @return int
   */
  int get_n() { return n; }
  /**
   * @brief セルの大きさを取得する
   *
   * @return float
   */
  float get_Lx() { return Lx; }
  /**
   * @brief Get the Ly object
   *
   * @return float
   */
  float get_Ly() { return Ly; }
  /**
   * @brief Get the Lz object
   *
   * @return float
   */
  float get_Lz() { return Lz; }
  virtual void load(const string &filename) = 0;  // モデルを読み込む
  virtual void save(const string &filename) const = 0;  // モデルを保存する
  /**
   * @brief Get the i last moved
   *
   * @return int
   */
  int get_i_last_moved() { return i_last_moved; }

 protected:
  float Lx, Ly, Lz;       // セルの大きさ[nm]
  int n;                  // 粒子数
  vector<X> coord;        // 各粒子の座標
  float cutoff;           // カットオフ距離(最接近距離)[nm]
  int i_last_moved = -1;  // 最後に動かした粒子のインデックス
  X last_move;            // 最後に動かした粒子の移動量
  int _n_iter = 0;        // 乱数を再生成した回数
  virtual bool is_in_cutoff(
      int i, int j) const = 0;  // 2粒子がカットオフ距離以内にあるかどうか
};

/**
 * @brief SAXSのデータを扱うクラス
 * 散乱強度Iは最大値が1になるように正規化する
 */
class SaxsData {
 public:
  inline SaxsData() {
    Q = Eigen::VectorXf(0);
    I = Eigen::VectorXf(0);
  }
  void load(const string &filename);
  void save(const string &filename) const;
  inline void setQ(const Eigen::VectorXf &q) { this->Q = q; }
  inline Eigen::VectorXf q() const { return this->Q; }
  inline void setI(const Eigen::VectorXf &I) { this->I = I / I.maxCoeff(); }
  inline Eigen::VectorXf i() const { return this->I; }
  inline unsigned long size() const { return this->Q.size(); }

 protected:
  Eigen::VectorXf Q;  // 波数[nm^-1]
  Eigen::VectorXf I;  // 散乱強度
};

template <typename M>
/**
 * @brief シミュレーターの基底クラス
 */
class Simulator {
 public:
  virtual void load_exp_data(
      const string &filename) = 0;  // 実験データを読み込む
  virtual int run(
      int n_step_max,
      double
          thresh) = 0;  // シミュレーションを実行して収束すれば0を、しなければ1を返す
  virtual void save(const std::string &filename) const = 0;  // 結果を保存
  inline int n() { return this->model.get_n(); }  // 粒子数を取得
  /**
   * @brief 最新の平均二乗誤差を取得
   *
   * @return float
   */
  inline double mse() { return this->mse_history.back(); }

 protected:
  M model;
  SaxsData exp_data;
  SaxsData sim_data;
  SaxsData sim_data_before;
  std::vector<double> mse_history{};  // TODO: 固定長のほうがいいかも
};

}  // namespace RmcCore

#endif  // RMC_CORE_HPP
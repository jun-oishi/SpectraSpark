#ifndef PLANAR_RMC_HPP
#define PLANAR_RMC_HPP

// #define EIGEN_NO_DEBUG
#include <Eigen/Dense>

#include "RmcCore.hpp"

namespace PlanarRmc {

/**
 * @brief 二次元三角格子上の粒子の配置を扱うモデル
 */
class TrianglePlanarModel : public RmcCore::Model<Eigen::Vector2i> {
 public:
  inline TrianglePlanarModel() { n = 0; };
  /**
   * @brief 粒子の配置を初期化する
   *
   * @param n  粒子数
   * @param Lx セルの大きさ(格子単位)
   * @param Ly セルの大きさ(格子単位)
   * @param a  格子定数
   * @param cutoff カットオフ距離
   */
  void init(int n, int Lx, int Ly, float a, float cutoff);
  /**
   * @brief 粒子をランダムに1つ選んで動かす
   */
  void move();

  /**
   * @brief 粒子の配置を1ステップ前に戻す
   */

  void undo();
  /**
   * @brief 2粒子の距離の二乗を計算する
   *
   * @param i
   * @param j
   * @return float
   */

  float dist2(int i, int j) const;
  /**
   * @brief 2粒子の距離を計算する
   *
   * @param i
   * @param j
   * @return float
   */
  float dist(int i, int j) const;
  /**
   * @brief Get the Lxi object
   *
   * @return int
   */
  int get_Lxi() const { return Lxi; };
  /**
   * @brief Get the Lyi object
   *
   * @return int
   */
  int get_Lyi() const { return Lyi; };
  /**
   * @brief 粒子の配置をcsvファイルから読み込む
   *
   * @param filename
   */
  void load(const std::string &filename);
  /**
   * @brief 粒子の配置をcsvファイルに保存する
   *
   * @param filename
   */
  void save(const std::string &filename) const;
  /**
   * @brief 粒子の実座標を取得する
   *
   * @param i
   * @return Eigen::Vector2f
   */
  Eigen::Vector2f real_coord(int i) const;

 protected:
  float a;       // 格子定数[nm]
  int Lxi, Lyi;  // セルの大きさ(格子単位)
  /**
   * @brief 2粒子がカットオフ距離以内にあればtrueを返す
   *
   * @param i
   * @param j
   * @return true
   * @return false
   */
  bool is_in_cutoff(int i, int j) const;
};

/**
 * @brief 二次元三角格子上の粒子の配置を扱うシミュレーター
 * 粒子は球とみなせるとしてSAXSのシミュレーションを行う
 */
class TrianglePlanarSimulator : public RmcCore::Simulator<TrianglePlanarModel> {
 public:
  inline TrianglePlanarSimulator(){};
  /**
   * @brief シミュレータと粒子配置を初期化する
   *
   * @param n
   * @param Lx
   * @param Ly
   * @param a
   * @param cutoff
   * @param r_par
   */
  void init(int n, int Lx, int Ly, float a, float cutoff, float r_par);
  /**
   * @brief 実験データを読み込む
   *
   * @param filename
   */
  void load_exp_data(const std::string &filename);
  /**
   * @brief シミュレーションを実行する
   *
   * @param n_step_max 最大ステップ数
   * @param thresh     収束判定のmseの閾値
   * @todo 収束判定の方法を見直す
   */
  int run(int n_step_max, double thresh);
  /**
   * @brief シミュレーションの結果を保存する
   */
  void save(const std::string &filename) const;

 protected:
  float r_par;  // 粒子の半径
  /**
   * @brief 粒子を動かして評価を行い必要ならば元に戻す
   */
  void move();
  /**
   * @brief モデルから散乱強度を計算してsim_dataに格納する
   * ref. Guinier abd Fournet, 1955, Small-angle Scattering of X-rays
   */
  void compute_i();
  std::vector<Eigen::MatrixXf>
      __re_exp_qr;  // exp(-iqr)の実部を格納:添字はq,theta,粒子iの順
  std::vector<Eigen::MatrixXf>
      __im_exp_qr;  // exp(-iqr)の虚部を格納:添字はq,theta,粒子iの順
  std::vector<double> __i_par;  // 粒子の散乱強度を格納
  /**
   * @brief exp(-iqr)を計算して__re_exp_qr, __im_exp_qrに格納する
   */
  void __compute_exp_qr(int i);
  /**
   * @brief 一粒子の散乱強度を計算する
   */
  void __compute_i_par();
};

}  // namespace PlanarRmc

#endif  // PLANAR_RMC_HPP
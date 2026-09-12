#ifndef AUTO_BUFF__TARGET_HPP
#define AUTO_BUFF__TARGET_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include "buff_detector.hpp"
#include "buff_type.hpp"
#include "tools/extended_kalman_filter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/ransac_sine_fitter.hpp"

namespace auto_buff
{
class Voter
{
public:
  Voter();
  void vote(const double angle_last, const double angle_now);
  int clockwise();

private:
  int clockwise_;
};

/// Target 基类

class Target
{
public:
  Target();
  virtual void get_target(
    const std::optional<PowerRune> & p,
    std::chrono::steady_clock::time_point & timestamp) = 0;  // 纯虚函数

  virtual void predict(double dt) = 0;  // 纯虚函数

  Eigen::Vector3d point_buff2world(const Eigen::Vector3d & point_in_buff) const;

  bool is_unsolve() const;

  Eigen::VectorXd ekf_x() const;

  // 诊断用: P 的对角。判"某个状态的不确定度是不是已经收到推不动了"需要它。
  // predict() 里 Q_ 只有左上 2×2 非零, R_pitch/R_dis/yaw/roll/spd 没有过程噪声,
  // 它们的 P 对角只能单调收缩 —— 收到接近 0 之后两级更新都推不动, init 那几帧留下的
  // 误差就被永久锁死。这个访问器让测试能直接量, 不必靠推理。
  Eigen::VectorXd ekf_P_diag() const;

  // 诊断开关, 默认 false = 生产行为。置 true 时 update() 跳过第二级(叶心)观测,
  // 用来把"阶段 2 每帧推歪中心"和"init 残留被零 Q_ 锁死"这两种机理分开。
  // 只由 tests/auto_buff_test.cpp 的 --no-stage2 设置, 生产代码从不碰它。
  bool diag_skip_stage2 = false;

  double spd = 0;  //调试用

protected:
  virtual void init(double nowtime, const PowerRune & p) = 0;  // 纯虚函数

  virtual void update(double nowtime, const PowerRune & p) = 0;  // 纯虚函数

  Eigen::VectorXd x0_;
  Eigen::MatrixXd P0_;
  Eigen::MatrixXd A_;
  Eigen::MatrixXd Q_;
  Eigen::MatrixXd H_;
  Eigen::MatrixXd R_;
  tools::ExtendedKalmanFilter ekf_;
  double lasttime_ = 0;
  Voter voter;  // 逆时针-1 顺时针1
  bool first_in_;
  bool unsolvable_;
};

/// SmallTarget子类

class SmallTarget : public Target
{
public:
  SmallTarget();

  void get_target(
    const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp) override;

  void predict(double dt) override;

private:
  void init(double nowtime, const PowerRune & p) override;

  void update(double nowtime, const PowerRune & p) override;

  Eigen::MatrixXd h_jacobian() const;

  const double SMALL_W = CV_PI / 3;
  // const double SMALL_W = 0;

  // spd 初值估计用的 roll 观测历史 (时间, roll)。
  // init() 时若历史足够, 用最近两对帧差分直接估转速 —— 真值 ~120 deg/s 与
  // 旧初值 60 相差一倍, EKF 从 60 爬到 120 要几十帧, 期间:
  //   绿框滞后(预测跟不上观测), 蓝框超前量逐帧膨胀(spd 在变, 蓝框不与绿框共速);
  //  画面上就是识别瞬间蓝框快速冲出。差分初值让两级更新从第一帧就接近真值。
  std::deque<std::pair<double, double>> roll_history_;
};

/// BigTarget子类

class BigTarget : public Target
{
public:
  BigTarget();

  void get_target(
    const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp) override;

  void predict(double dt) override;

private:
  void init(double nowtime, const PowerRune & p) override;

  void update(double nowtime, const PowerRune & p) override;

  Eigen::MatrixXd h_jacobian() const;

  tools::RansacSineFitter spd_fitter_;

  double fit_spd_;
};

}  // namespace auto_buff
#endif
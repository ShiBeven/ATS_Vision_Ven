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
  int clockwise() const;

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

/// BigTarget子类 —— 2026-09-13 重写: 单一 10 态 EKF 模型, RANSAC 降级为只读仪表。
/// 详见 buff_target.cpp BigTarget 段头注释与 buff_model/2026-09-13-大符整体方案设计.md。

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

  // spd 闭式: x[6] = a·sin(ωt+φ) + b —— b=x[10] 自由状态(官方数据收敛到 2.090−a; 匀速 a→0)
  double spd_at(const Eigen::VectorXd & x, double t) const;

  // roll 闭式积分(与状态定义严格一致, t0 = 最近观测时刻)
  double roll_at(const Eigen::VectorXd & x, double t0, double t) const;

  // roll(t2) − roll(t0) 对 x 的解析雅可比(1×11: x[5]/x[7]/x[8]/x[9]/x[10])
  Eigen::MatrixXd roll_diff_jacobian(const Eigen::VectorXd & x, double t0, double t2) const;

  // 会话边界/参数重置: 只重置 spd/a/ω/φ 子块, 保留 R 系与转子相位
  void reset_sine_params();

  void latch_predict_dt(double dt);

  tools::RansacSineFitter spd_fitter_;  // 只读仪表: 拟合结果不回流 predict

  int params_diverge_count_ = 0;

  // 第七轮: 1s 窗口相位跳变历史(时间戳, 是否发生槽位对齐), 用于跳变率触发
  // reset_sine_params —— 见 update() 中"跳变率检测"注释。
  std::deque<std::pair<double, bool>> jump_hist_;
  double last_predict_dt_ = 0.0;
  // 状态(x[5]/x[6])当前有效的时刻。predict() 推进状态后它 += dt;
  // update() 观测更新以它为积分起点, 之后 = nowtime。
  // 不能复用 lasttime_(上次观测时刻): aim 的前瞻 predict 会把状态推到
  // lasttime_+predict_dt, 闭式积分的 t0 必须跟着走, 否则相位基准错位。
  double phase_t0_ = 0.0;
  std::chrono::steady_clock::time_point start_timestamp_{};
  int lost_cn_ = 0;

  // ---- 第十八轮: R 标长期锚定(方案三) ----
  // 世界系静止点的会话级 EWMA 锚定值与观测计数, 详见 update() 内注释。
  double r_anchor_[3] = {0.0, 0.0, 0.0};
  int r_anchor_count_ = 0;
  
  // ---- 第二十三轮: 大符转向早期锁存(蓝框方向跳变修复) ----
  // Voter(基类)用原始角度大小比较投票: (1) 相位在 ±180deg 回绕处必投错票;
  // (2) 未到 50 票前 clockwise() 符号随票数过零翻转。大符转向开局已定,
  // 预测方向(roll_at 的 sgn)不该随后续抖动改变。这里用相邻观测相位差
  // (limit_rad 回绕安全)独立投票, 净票达 ±20 后永久锁定; 锁定后 roll_at/
  // roll_diff_jacobian 不再读 Voter。SmallTarget 的 Voter 路径零改动。
  int dir_votes_ = 0;
  bool dir_locked_ = false;
  int dir_sign_ = 0;  // 锁定时 ±1; 未锁定 0
  double last_phase_obs_ = 0.0;
  double last_phase_t_ = 0.0;
  bool has_last_phase_obs_ = false;

  // ---- 第二十四轮: 方案C 跳变后相位观测收紧 ----
  // 根因(日志实锤): d 单向爬升至+13~15deg 触发15deg槽位对齐硬跳, 锯齿周期
  // ≈正弦半周期; R1(3,3)=0.1(σ≈18deg) 太宽松, 13deg 残差增益不足。跳变后
  // 在窗口内收紧到 0.02(σ≈8deg) 让 EKF 快速咬住观测, 压掉锯齿峰值。
  // 窗口外恢复 0.1, 与原行为一致(关键点噪声大时不过度信任单帧)。
  int track_r1_tight_ = 0;

  // ---- 第十八轮: 丢失帧 keep-alive(方案六) ----
  // lasttime_ 是相对会话起始的秒数(double), keep-alive 的空窗判定需要
  // time_point 的 wall-clock —— 两者不同源, 这里单独维护。
  // now_tp_ 由 get_target 每有效观测帧透传, update() 再转记到 lasttime_tp_。
  std::chrono::steady_clock::time_point now_tp_{};
  std::chrono::steady_clock::time_point last_update_tp_{};
  std::chrono::steady_clock::time_point lasttime_tp_{};
  bool lasttime_tp_set_ = false;
};
}  // namespace auto_buff
#endif

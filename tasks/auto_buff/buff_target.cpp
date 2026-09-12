#include "buff_target.hpp"

#include <limits>

namespace auto_buff
{
///voter

Voter::Voter() : clockwise_(0) {}

void Voter::vote(const double angle_last, const double angle_now)
{
  if (std::abs(clockwise_) > 50) return;
  if (angle_last > angle_now)
    clockwise_--;
  else
    clockwise_++;
}

int Voter::clockwise() { return clockwise_ > 0 ? 1 : -1; }

/// Target

Target::Target() : first_in_(true), unsolvable_(true) {};

Eigen::Vector3d Target::point_buff2world(const Eigen::Vector3d & point_in_buff) const
{
  if (unsolvable_) return Eigen::Vector3d(0, 0, 0);
  Eigen::Matrix3d R_buff2world =
    tools::rotation_matrix(Eigen::Vector3d(ekf_.x[4], 0.0, ekf_.x[5]));  // pitch = 0

  auto R_yaw = ekf_.x[0];
  auto R_pitch = ekf_.x[2];
  auto R_dis = ekf_.x[3];
  Eigen::Vector3d point_in_world =
    R_buff2world * point_in_buff + Eigen::Vector3d(
                                     R_dis * std::cos(R_pitch) * std::cos(R_yaw),
                                     R_dis * std::cos(R_pitch) * std::sin(R_yaw),
                                     R_dis * std::sin(R_pitch));
  return point_in_world;
}

bool Target::is_unsolve() const { return unsolvable_; }

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

Eigen::VectorXd Target::ekf_P_diag() const
{
  if (ekf_.P.size() == 0) return Eigen::VectorXd();
  return ekf_.P.diagonal();
}

/// SmallTarget

SmallTarget::SmallTarget() : Target() {}

void SmallTarget::get_target(
  const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp)
{
  // 如果没有识别，退出函数
  static int lost_cn = 0;
  if (!p.has_value()) {
    unsolvable_ = true;
    lost_cn++;
    return;
  }

  static std::chrono::steady_clock::time_point start_timestamp = timestamp;
  auto time_gap = tools::delta_time(timestamp, start_timestamp);

  // init
  if (first_in_) {
    unsolvable_ = true;
    init(time_gap, p.value());
    first_in_ = false;
  }

  // 处理识别时间间隔过大
  if (lost_cn > 6) {
    unsolvable_ = true;
    tools::logger()->debug("[Target] 丢失buff");
    lost_cn = 0;
    first_in_ = true;
    return;
  }

  // kalman update
  unsolvable_ = false;
  update(time_gap, p.value());

  // 处理发散: 只防真离谱, 不再假设转速值。
  // 原门 [SMALL_W-10, SMALL_W+10] deg/s 把 "小符必为 60" 写死 ——
  // 录像真实 ~120 deg/s 时 spd 每次爬到 70 就被判发散踢回 60, 周期性重初始化,
  // 这就是绿框周期性跳变的另一半来源。转速真值应由滤波器从观测学出
  // (见 predict 里 spd 过程噪声), 不该由门预设。
  // 上界 4 rad/s (~229 deg/s) 防数值发散; 下界防转速贴零(EKF 已无跟踪能力)。
  if (std::abs(ekf_.x[6]) > 4.0 || std::abs(ekf_.x[6]) < CV_PI / 36) {
    unsolvable_ = true;
    tools::logger()->debug("[Target] 小符角度发散spd: {:.2f}", ekf_.x[6] * 180 / CV_PI);
    first_in_ = true;
    return;
  }
}

void SmallTarget::predict(double dt)
{
  // 预测下一个状态
  // clang-format off
  A_ << 1.0,  dt, 0.0, 0.0, 0.0, 0.0, 0.0, // R_yaw
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // R_v_yaw
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // R_pitch
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // R_dis
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // yaw
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0,  dt, // roll
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0; // spd

  // 过程噪声协方差矩阵                            //// 调整
  auto v1 = 0.001;  // 角加速度方差
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
    // spd 过程噪声: 转轴锁死后 roll 观测已干净, spd 完全可观。原先为 0,
  // P[6][6] 单调衰减, 增益消失, spd 冻死在初值 60, 而录像真实转速 ~120 deg/s,
  // 每帧残差 1.24 deg 累积成绿框滞后, 到 36 deg 触发 72 deg 整叶位跳变(超调到 +36),
  // 再落后再跳 —— 就是画面上落后-跳到叶上-循环的机理。
  // spd 过程噪声: 小符转速规则恒定(~60 deg/s), 只需慢速跟踪。
  // 曾用 0.25(sigma=29 deg/s), 结果关键点噪声逐帧推得动 spd,
  // spd 在 55~81 deg/s 振荡, 蓝框夹角(=spd*T)随之摆动十几度。
  // 0.01(sigma=5.7 deg/s): 单帧噪声推不动, 几十帧内仍能跟上真实漂移。
  constexpr double SPD_PROCESS_VAR = 0.01;  // (0.1 rad/s)^2
  Q_ << a * v1, b * v1, 0.0, 0.0, 0.0, 0.0, 0.0,
        b * v1, c * v1, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0,    0.0, 0.0, 0.0, 0.0, 0.0, SPD_PROCESS_VAR;
  // clang-format on 
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = A_ * x;
    x_prior[0] = tools::limit_rad(x_prior[0]);
    x_prior[2] = tools::limit_rad(x_prior[2]);
    x_prior[4] = tools::limit_rad(x_prior[4]);
    x_prior[5] = tools::limit_rad(x_prior[5]);
    return x_prior;
  };
  ekf_.predict(A_, Q_, f);
}

void SmallTarget::init(double nowtime, const PowerRune & p)
{
  // 初始化内部变量
  lasttime_ = nowtime;

  // 初始状态协方差矩阵
  x0_.resize(7);
  P0_.resize(7, 7);
  A_.resize(7, 7);
  Q_.resize(7, 7);
  H_.resize(7, 7);//z x
  R_.resize(7, 7);//z z
  // [R_yaw]
  // [v_R_yaw]
  // [R_pitch]
  // [R_dis]
  // [yaw]
  // [angle/row]
  // [spd]   w=CV_PI/6

  // clang-format off
  // 初始状态
  // spd 初值: 有足够的 roll 观测历史时用差分直接估计, 否则退回 SMALL_W。
  // 旧值恒为 SMALL_W(60 deg/s) — 录像真实 ~120 deg/s, EKF 要爬几十帧才收敛,
  // 期间绿框滞后、蓝框超前量逐帧膨胀(识别瞬间冲出的直接来源)。差分初值
  // 让两级更新从第一帧就贴近真值。回绕用 limit_rad 处理, 整叶位 72° 跳变
  // 用最近邻圈数对齐。
  double spd_init = SMALL_W * voter.clockwise();
  if (roll_history_.size() >= 2) {
    const auto & newest = roll_history_.back();
    const auto & oldest = roll_history_.front();
    double d_roll = 0.0;
    // 找与最新观测同叶位(72° 倍数)对齐后的角度差, 再 limit_rad 去回绕
    double best_err = std::numeric_limits<double>::max();
    for (int k = -5; k <= 5; k++) {
      const double cand = oldest.second + k * 2.0 * CV_PI / 5.0;
      const double err = std::fabs(tools::limit_rad(newest.second - cand));
      if (err < best_err) {
        best_err = err;
        d_roll = tools::limit_rad(newest.second - cand);
      }
    }
    const double dt_hist = newest.first - oldest.first;
    // 时间有效性: 历史太旧(丢失前残留)不用于差分, 退默认。0.5s 内视为有效。
    const double dt_stale = nowtime - newest.first;
    if (dt_hist > 1e-3 && dt_hist < 0.5 && dt_stale >= 0 && dt_stale < 0.5) {
      const double spd_diff = d_roll / dt_hist;
      // 只接受量级合理的差分(0.5~4 rad/s ≈ 29~229 deg/s), 荒谬值(断帧/换叶残余)退回默认
      if (std::fabs(spd_diff) > 0.5 && std::fabs(spd_diff) < 4.0) spd_init = spd_diff;
    }
  }
  x0_ << p.ypd_in_world[0], 0.0, p.ypd_in_world[1], p.ypd_in_world[2],
         p.ypr_in_world[0], p.ypr_in_world[2], 
         spd_init;
  // 初始状态协方差矩阵
  P0_ << 10.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0, 10.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0, 10.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0,  0.0, 10.0,  0.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0, 10.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0, 10.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1e-2;
  // 状态转移矩阵
  // A_ 
  // 过程噪声协方差矩阵                            //// 调整
  // Q_ 
  // 测量方程矩阵
  // H_
  // 测量噪声协方差矩阵                            //// 调整
  // R_

  // clang-format on

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[0] = tools::limit_rad(c[0]);
    c[2] = tools::limit_rad(c[2]);
    c[4] = tools::limit_rad(c[4]);
    c[5] = tools::limit_rad(c[5]);
    return c;
  };
  // 创建扩展卡尔曼滤波器对象
  ekf_ = tools::ExtendedKalmanFilter(x0_, P0_, x_add);
}

void SmallTarget::update(double nowtime, const PowerRune & p)
{
  // [R_yaw]     angle0
  // [v_R_yaw]
  // [R_pitch]   angle2
  // [R_dis]
  // [yaw]       angle4
  // [angle/row] angle5
  // [spd]   w=CV_PI/6
  const Eigen::VectorXd & R_ypd = p.ypd_in_world;  // R
  const Eigen::VectorXd & ypr = p.ypr_in_world;
  const Eigen::VectorXd & B_ypd = p.blade_ypd_in_world;  // center of blade

  // spd 初值估计: 记录 roll 观测历史 (init 里差分用, 回绕在差分时处理)。
  roll_history_.emplace_back(nowtime, ypr[2]);
  if (roll_history_.size() > 8) roll_history_.pop_front();

  // 处理扇叶跳变 angle/row
  // 【切板防污染】击中切板时 roll 观测跳 72°(换了一片叶), 这是叶位跳变, 不是转速信息。
  // EKF 的 P 里 roll-spd 有互协方差(predict 的 roll += dt*spd 每帧演化出耦合),
  // 大残差会经 K[6] 冲击 spd → 蓝框(超前角 = spd×T)在切板瞬间加速。
  // 对齐时同步切断 P[5][6]/P[6][5] 并放宽 P[5][5]: 叶位跳变不携带转速信息,
  // 转速只由后续正常帧的连续观测决定。
  bool blade_jump_aligned = false;
  if (abs(ypr[2] - ekf_.x[5]) > CV_PI / 12) {
    for (int i = -5; i <= 5; i++) {
      double angle_c = ekf_.x[5] + i * 2 * CV_PI / 5;
      if (std::fabs(angle_c - ypr[2]) < CV_PI / 5) {
        ekf_.x[5] += i * 2 * CV_PI / 5;
        blade_jump_aligned = true;
        break;
      }
    }
  }
  if (blade_jump_aligned) {
    // P[5][5] 放宽回初值量级(roll 位置刚跳过, 不确定度重置);
    // 互协方差清零(切断 roll 残差 → spd 的增益通道)。
    if (ekf_.P.size() > 0) {
      ekf_.P(5, 5) = 10.0;
      ekf_.P(5, 6) = 0.0;
      ekf_.P(6, 5) = 0.0;
    }
  }

  // vote判断是顺时针还是逆时针旋转
  voter.vote(ekf_.x[5], ypr[2]);
  if (voter.clockwise() * ekf_.x[6] < 0) ekf_.x[6] *= -1;  // spd

  // 预测下一个状态
  predict(nowtime - lasttime_);

  // [R_yaw]     angle0
  // [R_pitch]   angle1
  // [R_dis]
  // [angle/row] angle3
  // [B_yaw]     angle4
  // [B_pitch]   angle5
  // [B_dis]

  /// 1.

  // [R_yaw]     angle0
  // [R_pitch]   angle1
  // [R_dis]
  // [angle/row] angle3

  // clang-format off
  Eigen::MatrixXd H1{
    {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // R_yaw
    {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}, // R_pitch
    {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}, // R_dis
    {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}  // roll
  };

  // R1 是**方差**, 不是标准差。原先两个角度项都是 0.01 rad² → σ = 0.1 rad = 5.73°,
  // 即"转轴方向只准到 ±5.73°"。在 9.45 m 处这是 ±0.94 m, 按 fx=2367 折算 **±236 px**。
  //
  // 实测(800 帧, --fps=25): PnP 位姿把 (0,0,0) 投回来离检测到的 R 十字 median 3.0 px
  // = 0.073°。也就是说这个观测的实际精度比 R1 声称的**紧 79 倍**(方差差 6200 倍)。
  // 后果不是"滤波器出错", 而是滤波器在合法使用被给予的余量: 同一轮实测 EKF 状态
  // 投出来的转轴离 R 十字 median 134 px = 3.24°, 稳稳落在它以为的 ±5.73° 里。
  // 画面上就是绿/蓝两条线的转轴端离洋红十字一大截。
  //
  // 改成 σ = 5e-3 rad = 0.29° = 11.8 px → 方差 2.5e-5。刻意仍比实测残差松 4 倍:
  //   - 3.0 px 是 R 标进了点集之后的**样本内**残差, 不是独立精度;
  //   - OBJECT_POINTS 与实物尚有不一致(本轮 PnP RMS 从 0.17 升到 4.2 px, 见下面 R2 注释),
  //     这份几何误差也得有地方吸收。
  //
  // 【为什么不同时把 R2(叶心观测)的角度项一起收紧】不是漏了, 是故意的:
  // 阶段 1 的模型是 H1 那几行单位阵, 精确; 阶段 2 的 h2() 走 point_buff2world(),
  // 那里 pitch 被写死成 0, 而 PnP 解出的符面 pitch 未必为 0 —— 模型有结构性误差。
  // 模型误差在放不进状态时就该放进 R。两级同样紧只会让它们互相拉扯中心位置,
  // 而转轴的直接观测在阶段 1, 阶段 2 是经 700 mm 杠杆推出来的, 本就该让阶段 1 说话。
  Eigen::MatrixXd R1{
    {2.5e-5,    0.0, 0.0,  0.0}, // R_yaw   σ=0.29°(实测 0.073°)
    {   0.0, 2.5e-5, 0.0,  0.0}, // R_pitch σ=0.29°
    {   0.0,    0.0, 0.5,  0.0}, // R_dis   σ=0.71 m(实测 PnP std 0.40 m, 量级相符, 未动)
    {   0.0,    0.0, 0.0,  0.1}  // roll    σ=18°(未动: roll 归 spd 链路, 不在本次变量内)
  };
  // clang-format on

  // 防止夹角求差出现异常值
  auto z_subtract1 = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;  //4 1
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  Eigen::VectorXd z1{{R_ypd[0], R_ypd[1], R_ypd[2], ypr[2]}};  // R_ypd roll

  ekf_.update(z1, H1, R1, z_subtract1);

  ///2.

  // [B_yaw]     angle4
  // [B_pitch]   angle5
  // [B_dis]

  // diag_skip_stage2 默认 false, 即下面这一整级照原样执行 —— 生产行为逐位不变。
  // 置 true 只发生在 tests/auto_buff_test.cpp 传了 --no-stage2 时, 目的是把
  // "阶段 2 每帧把中心推歪" 和 "init 残留被零 Q_ 锁死" 这两种机理分开量。
  // 判据与副作用见那边 --no-stage2 那段注释。
  if (!diag_skip_stage2) {
    // clang-format off
    Eigen::MatrixXd H2 = h_jacobian();  // 3*7

    Eigen::MatrixXd R2{
      {0.01, 0.0, 0.0}, // B_yaw
      {0.0, 0.01, 0.0}, // B_pitch
      {0.0,  0.0, 0.5}  // B_dis
    };
    // clang-format on

    // 定义非线性转换函数h: x -> z
    auto h2 = [&](const Eigen::VectorXd & x) -> Eigen::Vector3d {
      Eigen::VectorXd R_ypd{{x[0], x[2], x[3]}};
      Eigen::VectorXd R_xyz = tools::ypd2xyz(R_ypd);
      Eigen::VectorXd R_xyz_and_yr{{R_ypd[0], R_ypd[1], R_ypd[2], x[4], x[5]}};
      Eigen::VectorXd B_xyz = point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
      Eigen::VectorXd B_ypd = tools::xyz2ypd(B_xyz);
      return B_ypd;
    };

    // 防止夹角求差出现异常值
    auto z_subtract2 = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
      Eigen::VectorXd c = a - b;  //6 1
      c[0] = tools::limit_rad(c[0]);
      c[1] = tools::limit_rad(c[1]);
      return c;
    };

    Eigen::VectorXd z2{{B_ypd[0], B_ypd[1], B_ypd[2]}};

    ekf_.update(z2, H2, R2, h2, z_subtract2);
  }

  // ================= R 标硬锁定: 转轴是全局定点, 不滤波 =================
  //
  // 【用户判据】绿/蓝框基准外延的那条线必须锁死在 R 标转轴上。R 标是全局定点,
  // 检测稳定位置固定 —— 它没有动力学过程需要滤波。实测两级更新后 EKF 的转轴状态
  // 偏离锚定观测 median 39 px, 且 v_R_yaw 被估出 max 61.7 deg/s 的"转轴角速度"
  // (转轴根本不动, 这是第二级叶心观测经 700mm 杠杆反灌进来的伪速度), 0.42s 的
  // predict 前推把它放大成上千像素的蓝线漂移。修法: 不再让 EKF 估计转轴,
  // 两级更新跑完后直接钳回锚定观测, v_R_yaw 恒 0。
  //
  // 【为什么可以整段钳掉】转轴状态 (R_yaw, v_R_yaw, R_pitch, R_dis) 的物理真值是
  // 常量(定点), EKF 对常量的最优估计就是观测本身 —— 而观测已被 R 像素锚定到
  // 1~3 px 精度(白线 median 3.0 px)。EKF 在这里没有增益, 只有被第二级拉扯的风险。
  //
  // 【为什么保留两级更新的形式】roll/spd/yaw(叶面自转)仍由两级更新估计, 只把转轴
  // 4 维从滤波范围里摘出来。--no-stage2 对照、P 对角仪表照常工作。
  {
    const auto & R_obs = p.ypd_in_world;
    ekf_.x[0] = R_obs[0];  // R_yaw
    ekf_.x[1] = 0.0;       // v_R_yaw —— 转轴不动, 没有角速度
    ekf_.x[2] = R_obs[1];  // R_pitch
    ekf_.x[3] = R_obs[2];  // R_dis
  }
  // 更新lasttime
  lasttime_ = nowtime;
  return;
}

Eigen::MatrixXd SmallTarget::h_jacobian() const
{
  /// Z(3,1) = H3(3,3) * H2(3,5) * H1(5,5) * H0(5,7) * x(7,1)

  // clang-format off
  Eigen::MatrixXd H0{
    {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}
  };// 5*7

  Eigen::VectorXd R_ypd{{ekf_.x[0], ekf_.x[2], ekf_.x[3]}};
  Eigen::MatrixXd H_ypd2xyz = tools::ypd2xyz_jacobian(R_ypd);  // 3*3
  Eigen::MatrixXd H1{
    {H_ypd2xyz(0, 0), H_ypd2xyz(0, 1), H_ypd2xyz(0, 2), 0.0, 0.0},
    {H_ypd2xyz(1, 0), H_ypd2xyz(1, 1), H_ypd2xyz(1, 2), 0.0, 0.0},
    {H_ypd2xyz(2, 0), H_ypd2xyz(2, 1), H_ypd2xyz(2, 2), 0.0, 0.0},
    {            0.0,             0.0,             0.0, 1.0, 0.0},
    {            0.0,             0.0,             0.0, 0.0, 1.0}
  };// 5*5

  // double pitch = 0;
  double yaw = ekf_.x[4];
  double roll = ekf_.x[5];
  double cos_yaw = cos(yaw);
  double sin_yaw = sin(yaw);
  double cos_roll = cos(roll);
  double sin_roll = sin(roll);
  Eigen::MatrixXd H2{
    {1.0, 0.0, 0.0, 0.7 * cos_yaw * sin_roll,  0.7 * sin_yaw * cos_roll},
    {0.0, 1.0, 0.0, 0.7 * sin_yaw * sin_roll, -0.7 * cos_yaw * cos_roll},
    {0.0, 0.0, 1.0,                      0.0,           -0.7 * sin_roll}
  };// 3*5

  Eigen::VectorXd B_xyz = point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
  Eigen::MatrixXd H3 = tools::xyz2ypd_jacobian(B_xyz);// 3*3
  // clang-format on

  return H3 * H2 * H1 * H0;  // 3*7

  // auto h2 = [&](const Eigen::VectorXd & x) -> Eigen::Vector3d {
  //   Eigen::VectorXd R_ypd{{x[0], x[2], x[3]}};
  //   Eigen::VectorXd R_xyz = tools::ypd2xyz(R_ypd);
  //   Eigen::VectorXd R_xyz_and_yr{{R_ypd[0], R_ypd[1], R_ypd[2], x[4], x[5]}};
  //   Eigen::VectorXd B_xyz = point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
  //   Eigen::VectorXd B_ypd = tools::xyz2ypd(B_xyz);
  //   return B_ypd;
  // };
}

/// BigTarget

BigTarget::BigTarget() : Target(), spd_fitter_(100, 0.5, 1.884, 2.000) {}

void BigTarget::get_target(
  const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp)
{
  // 如果没有识别，退出函数
  static int lost_cn = 0;
  if (!p.has_value()) {
    unsolvable_ = true;
    lost_cn++;
    return;
  }

  static std::chrono::steady_clock::time_point start_timestamp = timestamp;
  auto time_gap = tools::delta_time(timestamp, start_timestamp);

  // init
  if (first_in_) {
    unsolvable_ = true;
    init(time_gap, p.value());
    first_in_ = false;
  }

  // 处理识别时间间隔过大
  if (lost_cn > 6) {
    unsolvable_ = true;
    tools::logger()->debug("[Target] 丢失buff");
    lost_cn = 0;
    first_in_ = true;
    return;
  }

  // kalman update
  unsolvable_ = false;
  update(time_gap, p.value());

  // 处理发散
  if (
    ekf_.x[7] > 1.045 * 1.5 || ekf_.x[7] < 0.78 / 1.5 || ekf_.x[8] > 2.0 * 1.5 ||
    ekf_.x[8] < 1.884 / 1.5) {
    tools::logger()->debug("[Target] 大符角度发散a: {:.2f}b:{:.2f}", ekf_.x[7], ekf_.x[8]);
    first_in_ = true;
    return;
  }
}

void BigTarget::predict(double dt)
{
  // 预测下一个状态
  double spd = fit_spd_;
  // double spd = ekf_.x[6];
  double a = ekf_.x[7];
  double w = ekf_.x[8];
  double fi = ekf_.x[9];
  double t = lasttime_ + dt;
  // clang-format off
  A_ << 1.0,  dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,//R_yaw
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,//v_R_yaw
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,//R_pitch
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,//R_dis
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,//yaw
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, voter.clockwise() * dt , 0.0, 0.0, 0.0,//row
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, sin(w * t + fi) - 1, t * a * cos(w * t + fi), a * cos(w * t + fi),//spd
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,//a
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,//w
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;//theta
        
  // 过程噪声协方差矩阵                            //// 调整
  auto v1 = 0.9;  // 角加速度方差
  auto a1 = dt * dt * dt * dt / 4;
  auto b1 = dt * dt * dt / 2;
  auto c1 = dt * dt;
  Q_ << a1 * v1, b1 * v1, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
        b1 * v1, c1 * v1, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,     0.0, 0.0, 0.0, 0.0, 0.09,  0.0,  0.0,  0.0,  0.0,//row
            0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.5,  0.0,  0.0,  0.0,// spd 0.5  1
            0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,// a
            0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,// w
            0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  1.0;// fi
            // 0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  1.0,  0.0,  0.0,  0.0,// spd  2
            // 0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            // 0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 
            // 0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  4.0;

            // 0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            // 0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 
            // 0.0,     0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0;
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = x;
    x_prior[0] = tools::limit_rad(x_prior[0] + dt * x_prior[1]);
    x_prior[2] = tools::limit_rad(x_prior[2]);
    x_prior[4] = tools::limit_rad(x_prior[4]); // yaw
    x_prior[5] = tools::limit_rad(x_prior[5] + voter.clockwise() * 
    (-a / w * std::cos(w * t + fi) + a / w * std::cos(w * lasttime_ + fi) + (2.09 - a) * dt)); // roll
    x_prior[6] = a * sin(w * t + fi) + 2.09 - a; // spd
    return x_prior;
  };
  // clang-format on
  ekf_.predict(A_, Q_, f);
}

void BigTarget::init(double nowtime, const PowerRune & p)
{
  // 初始化内部变量
  lasttime_ = nowtime;
  unsolvable_ = true;

  // 初始状态协方差矩阵
  x0_.resize(10);
  P0_.resize(10, 10);
  A_.resize(10, 10);
  Q_.resize(10, 10);
  H_.resize(7, 10);
  R_.resize(7, 7);

  // [R_yaw]
  // [v_R_yaw]
  // [R_pitch]
  // [R_dis]
  // [yaw]
  // [angle/row]
  // [spd]       角速度 a*sin(wt) + 2.09 - a
  // [a]         0.78-1.045
  // [w]         1.884-2.000
  // [fi]

  // clang-format off
  // 初始状态
  x0_ << p.ypd_in_world[0], 0.0, p.ypd_in_world[1], p.ypd_in_world[2],
         p.ypr_in_world[0], p.ypr_in_world[2], 
         1.1775, 0.9125, 1.942, 0.0;//std::atan((spd - 2.09) / 0.9125 + 1
  // 初始状态协方差矩阵
  P0_ << 10.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0, 10.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0, 10.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0,  0.0, 10.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0, 10.0,  0.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0, 10.0,  0.0,  0.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 100.0, 0.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 10.0,  0.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 10.0,  0.0,
          0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 400.0;
  // 状态转移矩阵
  // A_
  // 过程噪声协方差矩阵                            //// 调整
  // Q_
  // 测量方程矩阵
  // H_
  // 测量噪声协方差矩阵                            //// 调整
  // R_

  // clang-format on

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[0] = tools::limit_rad(c[0]);
    c[2] = tools::limit_rad(c[2]);
    c[4] = tools::limit_rad(c[4]);
    c[5] = tools::limit_rad(c[5]);
    c[9] = tools::limit_rad(c[9]);
    return c;
  };
  // 创建扩展卡尔曼滤波器对象
  ekf_ = tools::ExtendedKalmanFilter(x0_, P0_, x_add);
}

void BigTarget::update(double nowtime, const PowerRune & p)
{
  // [R_yaw]
  // [v_R_yaw]
  // [R_pitch]
  // [R_dis]
  // [yaw]
  // [angle/row] 角度
  // [spd]       角速度 a*sin(wt) + 2.09 - a
  // [a]         0.78-1.045
  // [w]         1.884-2.000
  // [fi]
  const Eigen::VectorXd & R_ypd = p.ypd_in_world;  // R
  const Eigen::VectorXd & ypr = p.ypr_in_world;
  const Eigen::VectorXd & B_ypd = p.blade_ypd_in_world;  // center of blade

  // 处理扇叶跳变 angle/row
  // 【切板防污染】击中切板时 roll 观测跳 72°(换了一片叶), 这是叶位跳变, 不是转速信息。
  // EKF 的 P 里 roll-spd 有互协方差(predict 的 roll += dt*spd 每帧演化出耦合),
  // 大残差会经 K[6] 冲击 spd → 蓝框(超前角 = spd×T)在切板瞬间加速。
  // 对齐时同步切断 P[5][6]/P[6][5] 并放宽 P[5][5]: 叶位跳变不携带转速信息,
  // 转速只由后续正常帧的连续观测决定。
  bool blade_jump_aligned = false;
  if (abs(ypr[2] - ekf_.x[5]) > CV_PI / 12) {
    for (int i = -5; i <= 5; i++) {
      double angle_c = ekf_.x[5] + i * 2 * CV_PI / 5;
      if (std::fabs(angle_c - ypr[2]) < CV_PI / 5) {
        ekf_.x[5] += i * 2 * CV_PI / 5;
        blade_jump_aligned = true;
        break;
      }
    }
  }
  if (blade_jump_aligned) {
    // P[5][5] 放宽回初值量级(roll 位置刚跳过, 不确定度重置);
    // 互协方差清零(切断 roll 残差 → spd 的增益通道)。
    if (ekf_.P.size() > 0) {
      ekf_.P(5, 5) = 10.0;
      ekf_.P(5, 6) = 0.0;
      ekf_.P(6, 5) = 0.0;
    }
  }

  // vote判断是顺时针还是逆时针旋转
  voter.vote(ekf_.x[5], ypr[2]);

  auto anglelast = ekf_.x[5];  ///

  // 预测下一个状态
  predict(nowtime - lasttime_);

  // [R_yaw]     angle0
  // [R_pitch]   angle1
  // [R_dis]
  // [angle/row] angle3
  // [B_yaw]     angle4
  // [B_pitch]   angle5
  // [B_dis]

  /// 1.

  // [R_yaw]     angle0
  // [R_pitch]   angle1
  // [R_dis]
  // [angle/row] angle3

  // clang-format off
  Eigen::MatrixXd H1{
    {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // R_yaw
    {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // R_pitch
    {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // R_dis
    {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}  // roll
  };

  // 【已知缺陷, 故意暂未改】两个角度项 0.01 rad² = σ 5.73° 与 SmallTarget 那份是同一个,
  // 松了约 79 倍的理由见 SmallTarget::update() 里 R1 上方那段(带实测数字)。
  // 这里不跟着改, 因为手上只有小符录像, 改了无法验证 —— 等有大符录像跑出
  // 「转轴端离 R 十字」这条仪表再动, 别在没有判据的情况下改生产参数。
  Eigen::MatrixXd R1{
    {0.01, 0.0, 0.0,  0.0}, // R_yaw
    {0.0, 0.01, 0.0,  0.0}, // R_pitch
    {0.0,  0.0, 0.5,  0.0}, // R_dis
    {0.0,  0.0, 0.0, 0.1}  // roll  1: 0.01 2:0.04
  };
  // clang-format on

  // 防止夹角求差出现异常值
  auto z_subtract1 = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;  //4 1
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  Eigen::VectorXd z1{{R_ypd[0], R_ypd[1], R_ypd[2], ypr[2]}};  // R_ypd roll

  ekf_.update(z1, H1, R1, z_subtract1);

  ///2.

  // [B_yaw]     angle4
  // [B_pitch]   angle5
  // [B_dis]

  // clang-format off
  Eigen::MatrixXd H2 = h_jacobian();  // 3*10

  Eigen::MatrixXd R2{
    {0.01, 0.0, 0.0}, // B_yaw
    {0.0, 0.01, 0.0}, // B_pitch
    {0.0,  0.0, 0.5}  // B_dis
  };
  // clang-format on

  // 定义非线性转换函数h: x -> z
  auto h2 = [&](const Eigen::VectorXd & x) -> Eigen::Vector3d {
    Eigen::VectorXd R_ypd{{x[0], x[2], x[3]}};
    Eigen::VectorXd R_xyz = tools::ypd2xyz(R_ypd);
    Eigen::VectorXd R_xyz_and_yr{{R_ypd[0], R_ypd[1], R_ypd[2], x[4], x[5]}};
    Eigen::VectorXd B_xyz = point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
    Eigen::VectorXd B_ypd = tools::xyz2ypd(B_xyz);
    return B_ypd;
  };

  // 防止夹角求差出现异常值
  auto z_subtract2 = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;  //6 1
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    return c;
  };

  Eigen::VectorXd z2{{B_ypd[0], B_ypd[1], B_ypd[2]}};

  ekf_.update(z2, H2, R2, h2, z_subtract2);

  // ================= R 标硬锁定(同 SmallTarget, 理由见那边的注释) =================
  // 大符转轴同样是不动的定点, v_R_yaw 伪速度同样来自叶心观测反灌。
  {
    const auto & R_obs = p.ypd_in_world;
    ekf_.x[0] = R_obs[0];
    ekf_.x[1] = 0.0;
    ekf_.x[2] = R_obs[1];
    ekf_.x[3] = R_obs[2];
  }
  // 对ekf速度进行最小二乘拟合 ekf_.x[6] -> fitting_speed -> predict position
  if (ekf_.x[6] < 2.1 && ekf_.x[6] >= 0) spd_fitter_.add_data(nowtime, ekf_.x[6]);
  spd_fitter_.fit();

  fit_spd_ = spd_fitter_.sine_function(
    nowtime, spd_fitter_.best_result_.A, spd_fitter_.best_result_.omega,
    spd_fitter_.best_result_.phi, spd_fitter_.best_result_.C);

  spd = voter.clockwise() * (ekf_.x[5] - anglelast) / (nowtime - lasttime_);  // 仅供调试
  spd = fit_spd_;
  if (std::abs(spd) > 4) spd = 0;

  // 更新lasttime
  lasttime_ = nowtime;
  unsolvable_ = false;
  return;
}

Eigen::MatrixXd BigTarget::h_jacobian() const
{
  /// Z(3,1) = H3(3,3) * H2(3,5) * H1(5,5) * H0(5,10) * x(10,1)

  // clang-format off
  Eigen::MatrixXd H0{
    {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}
  };// 5*7

  Eigen::VectorXd R_ypd{{ekf_.x[0], ekf_.x[2], ekf_.x[3]}};
  Eigen::MatrixXd H_ypd2xyz = tools::ypd2xyz_jacobian(R_ypd);  // 3*3
  Eigen::MatrixXd H1{
    {H_ypd2xyz(0, 0), H_ypd2xyz(0, 1), H_ypd2xyz(0, 2), 0.0, 0.0},
    {H_ypd2xyz(1, 0), H_ypd2xyz(1, 1), H_ypd2xyz(1, 2), 0.0, 0.0},
    {H_ypd2xyz(2, 0), H_ypd2xyz(2, 1), H_ypd2xyz(2, 2), 0.0, 0.0},
    {            0.0,             0.0,             0.0, 1.0, 0.0},
    {            0.0,             0.0,             0.0, 0.0, 1.0}
  };// 5*5

  // double pitch = 0;
  double yaw = ekf_.x[4];
  double roll = ekf_.x[5];
  double cos_yaw = cos(yaw);
  double sin_yaw = sin(yaw);
  double cos_roll = cos(roll);
  double sin_roll = sin(roll);
  Eigen::MatrixXd H2{
    {1.0, 0.0, 0.0, 0.7 * cos_yaw * sin_roll,  0.7 * sin_yaw * cos_roll},
    {0.0, 1.0, 0.0, 0.7 * sin_yaw * sin_roll, -0.7 * cos_yaw * cos_roll},
    {0.0, 0.0, 1.0,                      0.0,           -0.7 * sin_roll}
  };// 3*5

  Eigen::VectorXd B_xyz = point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
  Eigen::MatrixXd H3 = tools::xyz2ypd_jacobian(B_xyz);// 3*3
  // clang-format on

  return H3 * H2 * H1 * H0;  // 3*7

  // auto h2 = [&](const Eigen::VectorXd & x) -> Eigen::Vector3d {
  //   Eigen::VectorXd R_ypd{{x[0], x[2], x[3]}};
  //   Eigen::VectorXd R_xyz = tools::ypd2xyz(R_ypd);
  //   Eigen::VectorXd R_xyz_and_yr{{R_ypd[0], R_ypd[1], R_ypd[2], x[4], x[5]}};
  //   Eigen::VectorXd B_xyz = point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
  //   Eigen::VectorXd B_ypd = tools::xyz2ypd(B_xyz);
  //   return B_ypd;
  // };
}
}  // namespace auto_buff

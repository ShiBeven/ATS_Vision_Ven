#ifndef AUTO_BUFF__AIMER_HPP
#define AUTO_BUFF__AIMER_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

#include "../auto_aim/planner/planner.hpp"
#include "buff_target.hpp"
#include "buff_type.hpp"
#include "io/command.hpp"
#include "io/gimbal/gimbal.hpp"

namespace auto_buff
{
class Aimer
{
public:
  Aimer(const std::string & config_path);

  io::Command aim(
    Target & target, std::chrono::steady_clock::time_point & timestamp, double bullet_speed,
    bool to_now = true);

  auto_aim::Plan mpc_aim(
    Target & target, std::chrono::steady_clock::time_point & timestamp, io::GimbalState gs,
    bool to_now = true);

  double angle;      ///
  double t_gap = 0;  ///

  // 调试/可视化用访问器: 画绿/蓝框的代码取同一份参数, 避免 yaml 双读不同步。
  double R_len() const { return R_len_; }
  double C_len() const { return C_len_; }

private:
  std::string config_path_;
  uint64_t runtime_params_version_ = 0;
  SmallTarget target_;
  double yaw_offset_;
  double pitch_offset_;

  // 绿框沿 R 轴圆周方向的人工微调参数(yaml: R_len / C_len), 与 yaw_offset/pitch_offset
  // 那种直接平移瞄准角的补偿是两种正交手段:
  //   - R_len: 缩放绿框底点(叶心, buff 系 z=0.7)到 R 标的距离, 1.0 = 不缩放;
  //   - C_len: 在圆周方向额外平移一个弧长(m), 正值超前(roll 增大方向, 即小符
  //     实际转向), 负值滞后。
  // 两参数只作用于"以 R 为圆心、叶面法向为切向"的圆弧方向, 不影响垂直该平面的分量,
  // 蓝框(预测)与绿框共用同一处 point_buff2world 调用, 自动跟随联动。
  double R_len_ = 1.0;
  double C_len_ = 0.0;

  double fire_gap_time_;
  double predict_time_;

  // 瞄准提前量缩放系数(yaml: buff_predict_lead_scale, 缺省 1.0 = 原行为)。
  // 作用于 aim() 里 future 的固定部分(通信/执行延迟补偿 + predict_time_),
  // 弹道飞行时间不受缩放(那是物理)。蓝框超前过多时调小, 0 = 无提前。
  double predict_lead_scale_ = 1.0;

  // MPC 速度/加速度前馈开关, 缺省 true(即原有行为)。
  // 验证识别层时置 false 只跑位置环, 免得前馈的已知缺陷把异常混进曲线导致无法归因。
  bool mpc_feedforward_ = true;

  int mistake_count_ = 0;
  // 补初始化: get_send_angle 首帧失败时下面的 if (switch_fanblade_) 会读到未初始化值(UB)。
  // false = 不在切扇叶, 与"刚启动还没判定过"的语义一致。
  bool switch_fanblade_ = false;

  double last_yaw_ = 0;
  double last_pitch_ = 0;

  // for mpc
  bool first_in_aimer_ = true;

  std::chrono::steady_clock::time_point last_fire_t_;

  void refresh_runtime_params_if_needed();
  bool get_send_angle(
    auto_buff::Target & target, const double predict_time, const double bullet_speed,
    const bool to_now, double & yaw, double & pitch);

  // R_len/C_len 应用后的实际瞄准叶心(世界系)。
  // 绿框重投影与瞄准求解共用这一定义, 保证可视化与打弹位置一致。
  Eigen::Vector3d aim_point_in_world(const Target & target) const;
};
}  // namespace auto_buff
#endif  // AUTO_AIM__AIMER_HPP

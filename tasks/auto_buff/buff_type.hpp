#ifndef BUFF__TYPE_HPP
#define BUFF__TYPE_HPP

#include <algorithm>
#include <deque>
#include <eigen3/Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

#include "tools/math_tools.hpp"
namespace auto_buff
{
const int INF = 1000000;
enum PowerRune_type { SMALL, BIG };
enum FanBlade_type { _target, _unlight, _light };
enum Track_status { TRACK, TEM_LOSE, LOSE };

class FanBlade
{
public:
  cv::Point2f center;               // 扇页中心
  std::vector<cv::Point2f> points;  // 四个点顺序为 上/左/下/右, 与 OBJECT_POINTS 前四点一致
  double angle, width, height;
  FanBlade_type type;  // 类型

  explicit FanBlade() = default;

  // explicit FanBlade(const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t);

  explicit FanBlade(
    const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t);

  explicit FanBlade(FanBlade_type t);
};

class PowerRune
{
public:
  cv::Point2f r_center;
  std::vector<FanBlade> fanblades;  // 按target开始顺时针

  int light_num;

  // 网络三分类: 0 未激活 / 1 小符已激活 / 2 大符已激活。仅透传至调试输出, 不参与任何判定
  int class_id = -1;

  Eigen::Vector3d xyz_in_world;  // 单位：m
  Eigen::Vector3d ypr_in_world;  // 单位：rad
  Eigen::Vector3d ypd_in_world;  // 球坐标系

  Eigen::Vector3d blade_xyz_in_world;  // 单位：m
  Eigen::Vector3d blade_ypd_in_world;  // 球坐标系, 单位: m

  // 几何相位（第七轮新增）：叶心方向相对世界竖直方向的转角，直接由 3D 向量几何计算。
  // 为什么不用 ypr_in_world[2]：欧拉分解在 IPPE 双解翻转时会把 yaw/pitch 的跳变泄漏进 roll，
  // 造成观测转速隔帧出现 -132~-312 deg/s 的物理不可能值（2026-09-13 18:32 日志已证实）。
  // 本字段在 Solver::solve() 末尾用 R_buff2world 的列向量 + 世界竖直向量直接计算，无奇异性。
  double blade_phase = 0.0;  // 单位: rad

  explicit PowerRune(
    std::vector<FanBlade> & ts, const cv::Point2f r_center,
    std::optional<PowerRune> last_powerrune, int class_id = -1);
  explicit PowerRune() = default;

  FanBlade & target() { return fanblades[0]; };

  bool is_unsolve() const { return unsolvable_; }

private:
  double target_angle_;
  bool unsolvable_ = false;

  double atan_angle(cv::Point2f v) const;  // [0, 2CV_PI]
};
}  // namespace auto_buff
#endif  // BUFF_TYPE_HPP

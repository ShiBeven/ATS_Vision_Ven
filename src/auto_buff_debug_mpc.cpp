#include <fmt/format.h>

#include <cmath>
#include <string>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/trajectory.hpp"

// 定义命令行参数
const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   |  configs/standard3.yaml | yaml配置文件路径 }"
  "{target        | small | 目标类型: small=小符(detect), big=大符(detect_big) }";

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  const std::string target_type = cli.get<std::string>("target");
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }
  if (target_type != "small" && target_type != "big") {
    tools::logger()->error("--target 只能是 small 或 big, 收到: {}", target_type);
    return 1;
  }
  const bool use_big_detect = (target_type == "big");

  // 初始化绘图器、录制器、退出器
  tools::Plotter plotter;
  tools::Recorder recorder;
  tools::Exiter exiter;

  // 初始化云台、相机
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  // 初始化识别器、解算器、追踪器、瞄准器
  auto_buff::Buff_Detector detector(config_path);
  auto_buff::Solver solver(config_path);
  auto_buff::SmallTarget small_target;
  auto_buff::BigTarget big_target;
  auto_buff::Aimer aimer(config_path);

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  while (!exiter.exit()) {
    camera.read(img, t);
    q = gimbal.q(t);
    auto gs = gimbal.state();
    // recorder.record(img, q, t);

    // -------------- 打符核心逻辑 --------------

    solver.set_R_gimbal2world(q);

    // 小符: 原 detect() 单候选, 行为零改动; 大符: detect_big() 多候选+身份锁定
    // (与测试链路同一套)。phase_hint 当前不参与选叶(接口保留), 传上一帧 EKF
    // 转子相位, 未解算时 nullopt。
    std::optional<auto_buff::PowerRune> power_runes;
    if (use_big_detect) {
      std::optional<double> phase_hint;
      if (!big_target.is_unsolve() && big_target.ekf_x().size() > 5)
        phase_hint = big_target.ekf_x()[5];
      power_runes = detector.detect_big(img, phase_hint);
    } else {
      power_runes = detector.detect(img);
    }

    solver.solve(power_runes);

    // 统一走基类指针: 小符恒 SmallTarget, 大符恒 BigTarget, EKF 状态由各自对象持有
    auto_buff::Target * target =
      use_big_detect ? static_cast<auto_buff::Target *>(&big_target)
                     : static_cast<auto_buff::Target *>(&small_target);
    target->get_target(power_runes, t);

    // 副本化(与测试链路同构): aim 的前瞻 predict 只推副本, 真身 EKF 不吃 predict。
    // 三目两分支类型不同不能直接推 auto, 故各留一份副本, 用基类指针统一喂给 aimer。
    auto small_copy = small_target;
    auto big_copy = big_target;
    auto_buff::Target * target_copy =
      use_big_detect ? static_cast<auto_buff::Target *>(&big_copy)
                     : static_cast<auto_buff::Target *>(&small_copy);

    auto plan = aimer.mpc_aim(*target_copy, t, gs, true);

    gimbal.send(
      plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
      plan.pitch_acc);
    // -------------- 调试输出 --------------

    nlohmann::json data;

    // buff原始观测数据
    if (power_runes.has_value()) {
      const auto & p = power_runes.value();
      data["buff_R_yaw"] = p.ypd_in_world[0];
      data["buff_R_pitch"] = p.ypd_in_world[1];
      data["buff_R_dis"] = p.ypd_in_world[2];
      data["buff_yaw"] = p.ypr_in_world[0] * 57.3;
      data["buff_pitch"] = p.ypr_in_world[1] * 57.3;
      data["buff_roll"] = p.ypr_in_world[2] * 57.3;
      data["buff_class_id"] = p.class_id;  // 0 未激活 / 1 小符已激活 / 2 大符已激活, 仅观察用
    }

    if (!target->is_unsolve()) {
      auto & p = power_runes.value();

      // 显示
      for (int i = 0; i < 4; i++) tools::draw_point(img, p.target().points[i]);
      tools::draw_point(img, p.target().center, {0, 0, 255}, 3);
      tools::draw_point(img, p.r_center, {0, 0, 255}, 3);

      // ---- 分层可视化(与 tests/auto_buff_test.cpp 同一套约定) ----
      //   洋红十字  检测器直出的 R 标 —— 一切判断的基准, 转轴真值
      //   白色线   纯 PnP 位姿的 叶心→转轴 线 —— 几何层
      //   绿色线框 当前帧 EKF 状态重投影 —— 滤波层
      //   蓝色线框 预测(超前)位置重投影 —— 预测层
      // 三条轴线的端点都应落在洋红十字上: 白线偏 → PnP/几何;
      // 白线准而绿线偏 → EKF; 绿线准而蓝线乱 → 预测/spd。

      // 白色 = 纯 PnP 位姿, 不经过 EKF
      const cv::Point2f pnp_blade = solver.point_buff2pixel(cv::Point3f(0, 0, 700e-3));
      const cv::Point2f pnp_axis = solver.point_buff2pixel(cv::Point3f(0, 0, 0));
      cv::line(img, pnp_blade, pnp_axis, {255, 255, 255}, 1, cv::LINE_AA);
      cv::circle(img, pnp_axis, 6, {255, 255, 255}, 1, cv::LINE_AA);

      // 绿色 = 当前帧 EKF 状态重投影, 应当贴合扇叶
      // R_len/C_len 在这里生效: 径向缩放 + 圆周偏移, 轴端仍钉在 R 标上,
      // 与瞄准点(buff_aimer.cpp 的 aim_point_in_world)是同一套参数。
      auto Rxyz_in_world_now = target->point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.0));
      auto image_points = solver.reproject_buff(
        Rxyz_in_world_now, target->ekf_x()[4], target->ekf_x()[5], aimer.R_len(), aimer.C_len());
      tools::draw_points(
        img, std::vector<cv::Point2f>(image_points.begin(), image_points.begin() + 4), {0, 255, 0});
      cv::line(img, image_points[4], image_points[6], {0, 255, 0}, 1, cv::LINE_AA);
      cv::circle(img, image_points[6], 6, {0, 255, 0}, 1, cv::LINE_AA);
      const cv::Point2f green_blade = image_points[4];

      // 蓝色 = 预测(超前)位置重投影, 应当沿旋转方向领先绿框
      auto Rxyz_in_world_pre = target_copy->point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.0));
      image_points = solver.reproject_buff(
        Rxyz_in_world_pre, target_copy->ekf_x()[4], target_copy->ekf_x()[5], aimer.R_len(),
        aimer.C_len());
      tools::draw_points(
        img, std::vector<cv::Point2f>(image_points.begin(), image_points.begin() + 4), {255, 0, 0});
      cv::line(img, image_points[4], image_points[6], {255, 0, 0}, 1, cv::LINE_AA);
      const cv::Point2f blue_blade = image_points[4];

      // 绿蓝夹角: 以 R 十字为顶点, 蓝框叶心相对绿框叶心的角度。
      // 共速判据 —— 夹角恒定 ⟺ 绿蓝共速; 突变说明 spd 被冲击。
      {
        const cv::Point2f rv = p.r_center;
        const auto angle_at = [&rv](const cv::Point2f & pt) {
          return std::atan2(static_cast<double>(pt.y - rv.y), static_cast<double>(pt.x - rv.x));
        };
        auto wrap_deg = [](double rad) {
          double d = rad * 57.29577951308232;
          while (d > 180.0) d -= 360.0;
          while (d < -180.0) d += 360.0;
          return d;
        };
        data["green_blue_angle_deg"] = wrap_deg(angle_at(blue_blade) - angle_at(green_blade));
        data["green_lag_deg"] = wrap_deg(angle_at(p.target().center) - angle_at(green_blade));
      }

      // 观测器内部数据
      Eigen::VectorXd x = target->ekf_x();
      data["R_yaw"] = x[0];
      data["R_V_yaw"] = x[1];
      data["R_pitch"] = x[2];
      data["R_dis"] = x[3];
      data["yaw"] = x[4] * 57.3;

      data["angle"] = x[5] * 57.3;
      data["spd"] = x[6] * 57.3;
      if (x.size() >= 11) {  // 大符 11 态: spd = a·sin(ωt+φ) + b, b=x[10]
        data["spd"] = x[6];
        data["a"] = x[7];
        data["w"] = x[8];
        data["fi"] = x[9];
        data["b"] = x[10];
        data["spd0"] = target->spd;
      }
    }

    // 云台响应情况
    data["gimbal_yaw"] = gs.yaw * 57.3;
    data["gimbal_pitch"] = gs.pitch * 57.3;
    data["gimbal_yaw_vel"] = gs.yaw_vel * 57.3;
    data["gimbal_pitch_vel"] = gs.pitch_vel * 57.3;

    if (plan.control) {
      data["plan_yaw"] = plan.yaw * 57.3;
      data["plan_pitch"] = plan.pitch * 57.3;
      data["plan_yaw_vel"] = plan.yaw_vel * 57.3;
      data["plan_pitch_vel"] = plan.pitch_vel * 57.3;
      data["plan_yaw_acc"] = plan.yaw_acc * 57.3;
      data["plan_pitch_acc"] = plan.pitch_acc * 57.3;
      data["shoot"] = plan.fire ? 1 : 0;
    }

    plotter.plot(data);

    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("result", img);

    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  return 0;
}
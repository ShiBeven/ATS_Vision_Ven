// 大符录像采集工具（第十轮配套）。
//
// 背景: 现有 符.avi / 符2.avi 实测含 29%~33% 的 H.264 skip 帧（VFR→CFR 转码填充），
// 观测转速被污染成 ≈0°/s 与 2~3 倍真值的交替（见实施记录第八/九轮）。本工具从
// 相机直采写盘，从源头杜绝该问题。
//
// 设计要点（为什么这样写）:
// 1. 编码用 MJPG（帧内压缩, 无帧间预测）—— 每帧独立编码, 不存在"复制上帧"的
//    skip 帧；体积约为 H.264 的 3~5 倍, 但对验证用途无所谓。
// 2. 帧率以实测为准: 用相机帧间隔的中位数标定, 不信容器头。同时实时显示
//    采集帧率, 录完输出帧间隔统计(抖动/P95), 供判断采集链路是否健康。
// 3. 四元数日志同步: 每帧把云台四元数(wxyz)与相对时间写入同名 .txt,
//    与 auto_buff_test 的日志格式一致（text >> t >> w >> x >> y >> z）。
// 4. 录完自动自检: 对写盘文件做相邻帧差分, 输出近零簇占比 —— 录完当场
//    就知道录像干不干净, 不用等跑完 auto_buff_test 才发现。
//
// 用法:
//   ./build/buff_record configs/standard3.yaml --output=assets/demo/符3 --seconds=60
//   （生成 符3.avi + 符3.txt；按 q 提前结束）

#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "io/gimbal/gimbal.hpp"
#include "io/camera.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/path.hpp"

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明                    }"
  "{seconds s      | 60                     | 录制时长(秒), 0=不限时按q结束          }"
  "{output o       | assets/demo/符3        | 输出文件名(不带后缀, 生成 .avi/.txt)   }"
  "{@config-path   | configs/standard3.yaml | yaml配置文件的路径                    }";

// 帧间隔统计: 中位数标定帧率 + P95 抖动, 判断采集链路健康度。
struct IntervalStats
{
  double median_ms = 0.0;
  double p95_ms = 0.0;
  double max_ms = 0.0;
  double fps = 0.0;
};

IntervalStats analyze_intervals(const std::vector<double> & intervals_ms)
{
  IntervalStats st;
  if (intervals_ms.empty()) return st;
  std::vector<double> sorted = intervals_ms;
  std::sort(sorted.begin(), sorted.end());
  const auto q = [&](double p) {
    const size_t idx = std::min(sorted.size() - 1, static_cast<size_t>(p * sorted.size()));
    return sorted[idx];
  };
  st.median_ms = q(0.5);
  st.p95_ms = q(0.95);
  st.max_ms = sorted.back();
  st.fps = 1000.0 / st.median_ms;
  return st;
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  const auto config_path = cli.get<std::string>(0);
  const auto output_base = cli.get<std::string>("output");
  const auto seconds = cli.get<double>("seconds");

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  // 先读一帧拿分辨率
  cv::Mat img;
  std::chrono::steady_clock::time_point ts;
  camera.read(img, ts);
  if (img.empty()) {
    tools::logger()->error("相机读取失败, 检查配置与连线");
    return 1;
  }
  const int w = img.cols, h = img.rows;

  // 输出路径处理: 相对路径落到当前执行目录（与 capture.cpp 同策略）
  const auto base_path = std::filesystem::path(output_base).is_absolute() ?
                         std::filesystem::path(output_base) :
                         std::filesystem::absolute(output_base);
  std::filesystem::create_directories(base_path.parent_path());
  const std::string avi_path = base_path.string() + ".avi";
  const std::string txt_path = base_path.string() + ".txt";

  // MJPG: 帧内压缩。H.264 的帧间预测在低码率/转码场景会产生 skip 帧
  // (这正是符.avi 的病因), MJPG 从编码层面排除该可能性。
  cv::VideoWriter writer(avi_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0,
                         cv::Size(w, h));
  if (!writer.isOpened()) {
    tools::logger()->error("无法创建输出文件: {}", avi_path);
    return 1;
  }
  std::ofstream q_log(txt_path);
  if (!q_log.is_open()) {
    tools::logger()->error("无法创建四元数日志: {}", txt_path);
    return 1;
  }

  tools::logger()->info(
    "开始录制: {} ({}x{}), 时限 {:.0f}s, 按 q 停止", avi_path, w, h, seconds);

  const auto t_start = std::chrono::steady_clock::now();
  std::vector<double> intervals_ms;
  std::chrono::steady_clock::time_point ts_prev = ts;
  long frame_cnt = 0;
  bool q_pressed = false;

  while (true) {
    camera.read(img, ts);
    if (img.empty()) break;

    // 相对时间戳(秒) —— 与 auto_buff_test 的 text 日志格式一致
    const double t_rel =
      std::chrono::duration<double>(ts - t_start).count();

    // 帧间隔统计(仅统计首帧之后)
    if (frame_cnt > 0) {
      const double dt_ms =
        std::chrono::duration<double, std::milli>(ts - ts_prev).count();
      intervals_ms.push_back(dt_ms);
    }
    ts_prev = ts;

    // 写视频帧 + 四元数日志(wxyz 顺序, 与 capture.cpp 一致)
    writer.write(img);
    const Eigen::Quaterniond q = gimbal.q(ts);
    q_log << fmt::format("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f}", t_rel, q.w(), q.x(), q.y(),
                         q.z()) << '\n';

    frame_cnt++;

    // 显示 + 按键
    cv::Mat show;
    cv::resize(img, show, {}, 0.6, 0.6);
    // 实时叠加: 帧号/已录时长/实时帧率
    const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    const double inst_fps = frame_cnt / std::max(elapsed, 1e-6);
    tools::draw_text(show, fmt::format("frame {}  {:.1f}s  {:.1f} fps", frame_cnt, elapsed,
                                       inst_fps), {10, 25}, {0, 255, 0});
    cv::imshow("buff_record (q=stop)", show);
    const int key = cv::waitKey(1);
    if (key == 'q') {
      q_pressed = true;
      break;
    }
    if (seconds > 0 && elapsed >= seconds) break;
  }

  writer.release();
  q_log.close();
  cv::destroyAllWindows();

  // ---- 采集健康度统计 ----
  const auto st = analyze_intervals(intervals_ms);
  tools::logger()->info(
    "[采集统计] 帧数 {} | 实测帧率 {:.2f} fps (中位间隔 {:.1f} ms) | P95 {:.1f} ms | "
    "最大 {:.1f} ms",
    frame_cnt, st.fps, st.median_ms, st.p95_ms, st.max_ms);
  if (frame_cnt > 0) {
    tools::logger()->info("[输出] {} + {}", avi_path, txt_path);
  }

  // ---- 录后自检: 相邻帧差分近零簇占比 ----
  // (与 auto_buff_test 的判重自检同口径: 解码刚写盘的文件, 差分<0.5 视为近零)
  {
    cv::VideoCapture check(avi_path);
    cv::Mat f0, f1;
    std::vector<double> diffs;
    while (check.read(f1)) {
      if (!f0.empty()) {
        cv::Mat g0, g1;
        cv::cvtColor(f0, g0, cv::COLOR_BGR2GRAY);
        cv::cvtColor(f1, g1, cv::COLOR_BGR2GRAY);
        cv::resize(g0, g0, {}, 0.25, 0.25);
        cv::resize(g1, g1, {}, 0.25, 0.25);
        cv::Mat d;
        cv::absdiff(g0, g1, d);
        diffs.push_back(cv::mean(d)[0]);
      }
      f0 = f1.clone();
    }
    if (diffs.size() >= 10) {
      size_t near_zero = 0;
      for (double d : diffs)
        if (d < 0.5) near_zero++;
      const double ratio = 100.0 * static_cast<double>(near_zero) / static_cast<double>(diffs.size());
      if (ratio < 10.0) {
        tools::logger()->info(
          "[录后自检] 相邻帧差分: 近零帧 {}/{} ({:.1f}%) < 10% —— 录像健康, 可用于验证",
          near_zero, diffs.size(), ratio);
      } else {
        tools::logger()->warn(
          "[录后自检] 相邻帧差分: 近零帧 {}/{} ({:.1f}%) ≥ 10% —— 录像疑似含重复帧, "
          "检查相机是否掉帧/带宽不足",
          near_zero, diffs.size(), ratio);
      }
    }
  }

  if (q_pressed) tools::logger()->info("已按 q 停止");
  return 0;
}
// 打符检测层的离线验证入口。
//
// 只跑 Buff_Detector（模型推理 + 解码 + PowerRune 组装），不做 PnP / EKF / 云台，
// 因此不需要四元数日志。现有的 auto_buff_test 要求 .avi 与同名 .txt 配对，
// 而模型库自带的 测试视频/符.avi 没有配套 .txt，用不了，故有此入口。
//
// 主要用来肉眼确认三件事：
//   1. R 标（洋红点，标注 R）是否稳定贴在能量机关中心标志上
//   2. 四角点序号是否为 1 上 → 2 左 → 3 下 → 4 右（接反会导致 PnP 镜像姿态）
//   3. 类别名是否随符的激活状态正确切换
// 绘制由 YOLO11_BUFF::draw() 直接画在输入图上。

#include <fmt/core.h>

#include <opencv2/opencv.hpp>

#include "tasks/auto_buff/buff_detector.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/path.hpp"

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明        }"
  "{@config-path   | configs/standard3.yaml | yaml配置文件的路径        }"
  "{@video-path    | assets/demo/符.avi     | 待检测视频的路径          }"
  "{save           |                        | 结果另存为 avi 的路径     }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  auto config_path = cli.get<std::string>(0);
  auto video_path = cli.get<std::string>(1);
  auto save_path = cli.get<std::string>("save");

  if (video_path.empty()) {
    tools::logger()->error("未指定视频路径! 用法: buff_detect_video_test <config> <video>");
    cli.printMessage();
    return 1;
  }

  // 与 auto_buff_test 一致地解析录像路径，这样从 build/ 目录启动也能找到 assets/ 下的视频
  video_path = tools::resolve_runtime_path_string(video_path);

  cv::VideoCapture video(video_path);
  if (!video.isOpened()) {
    tools::logger()->error("无法打开视频: {}", video_path);
    return 1;
  }
  tools::logger()->info("视频: {}", video_path);

  tools::Exiter exiter;
  auto_buff::Buff_Detector detector(config_path);

  cv::VideoWriter writer;
  cv::Mat img;
  bool size_logged = false;

  while (!exiter.exit()) {
    video.read(img);
    if (img.empty()) break;

    // 打印一次实际帧尺寸，用来判断预处理是否走了 padding 分支
    if (!size_logged) {
      tools::logger()->info("视频帧尺寸: {}x{} (模型输入 640x480)", img.cols, img.rows);
      size_logged = true;
    }

    auto power_runes = detector.detect(img);

    if (power_runes.has_value()) {
      const auto & p = power_runes.value();
      // 检测层绘制已由 YOLO11_BUFF::draw() 完成，这里补画进入 PowerRune 的 r_center，
      // 用来确认 R 标确实透传到了数据层
      tools::draw_point(img, p.r_center, {0, 0, 255}, 3);
      cv::putText(
        img, fmt::format("class_id: {}  light_num: {}", p.class_id, p.light_num),
        cv::Point(20, 80), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 255, 0), 2);
    } else {
      cv::putText(img, "no detection", cv::Point(20, 80), cv::FONT_HERSHEY_PLAIN, 2.0,
                  cv::Scalar(0, 0, 255), 2);
    }

    if (!save_path.empty()) {
      if (!writer.isOpened())
        writer.open(
          save_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0, img.size());
      writer.write(img);
    }

    cv::Mat show;
    cv::resize(img, show, {}, 0.7, 0.7);
    cv::imshow("buff_detect", show);

    // q 退出，空格暂停/继续
    int key = cv::waitKey(30);
    if (key == 'q') break;
    while (key == ' ') {
      int k = cv::waitKey(30);
      if (k == ' ' || k == 'q') {
        key = k;
        break;
      }
    }
    if (key == 'q') break;
  }

  if (writer.isOpened()) writer.release();
  cv::destroyAllWindows();
  return 0;
}

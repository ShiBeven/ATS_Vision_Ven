#ifndef AUTO_BUFF__YOLO11_BUFF_HPP
#define AUTO_BUFF__YOLO11_BUFF_HPP

// 注意：类名与文件名保留 YOLO11_BUFF / yolo11_buff.* 只是为了不牵动引用面，
// 实际承载的是模型库的 Rune-v8n-fp16-20260624 网络：
//   输入 1x3x480x640(NCHW)，输出 [1, 18, 6300]
//   18 = 3(类别分数) + 5(关键点) * 3(x, y, conf)
//   关键点顺序为 top, left, R, right, bottom —— 索引 2 是 R 标
// 解码语义对照模型库 RunePostProcessor::postProcessRuneMat 移植。

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "tools/logger.hpp"

namespace auto_buff
{
// 对应模型库 network_postprocess.hpp:91 的 {"未激活", "小符已激活", "大符已激活"}。
// 这里用 ASCII 是因为 cv::putText 画不出中文。
const std::vector<std::string> class_names = {"inactive", "small_activated", "big_activated"};

class YOLO11_BUFF
{
public:
  struct Object
  {
    std::vector<cv::Point2f> kpt;  // 4 角点，已重排为 上/左/下/右
    cv::Point2f r;                 // R 标（网络直出）
    cv::Point2f center;            // 四角点均值，补旧模型的扇叶中心
    int class_id;                  // 0 未激活 / 1 小符已激活 / 2 大符已激活
    float prob;
  };

  // 单帧解码的候选统计, 只读, 供离线诊断用 —— 记录过程不改变任何检测行为。
  //
  // 为什么需要: detect() 走 get_onecandidatebox(), 只把最高分候选交给上层,
  // 于是"这一帧其实有几个候选、选中的是哪一片叶"在上层完全不可见。
  // 而新模型把 5 个扇叶各自作为独立候选输出, "谁分最高"会在帧间变动,
  // 帧间换叶恰恰藏在这个不可见的地方。
  struct DecodeStats
  {
    int candidate_count = 0;     // centerDistanceNMS 之后的候选数
    std::vector<int> class_ids;  // 每个候选的 class_id, 按置信度降序
    int chosen_class_id = -1;    // 实际交给上层的那个候选
    cv::Point2f chosen_center{-1.f, -1.f};
    float chosen_prob = 0.f;
  };

  YOLO11_BUFF(const std::string & config);

  // 返回全部通过 centerDistanceNMS 的候选
  std::vector<Object> get_multicandidateboxes(cv::Mat & image);

  // 只返回置信度最高的一个候选
  std::vector<Object> get_onecandidatebox(cv::Mat & image);

  const DecodeStats & last_stats() const { return stats_; }

private:
  std::string device_;
  ov::Core core;  // 创建OpenVINO Runtime Core对象
  std::shared_ptr<ov::Model> model;
  ov::CompiledModel compiled_model;
  ov::InferRequest infer_request;
  ov::Tensor input_tensor;

  static constexpr int NUM_POINTS = 5;   // 网络输出的关键点数（含 R 标）
  static constexpr int NUM_CLASSES = 3;  // 三分类
  static constexpr int INPUT_H = 480;
  static constexpr int INPUT_W = 640;

  // 预处理的逆变换参数，解码时用来把关键点还原回原图坐标
  float scale_ = 1.0f;
  int pad_x_ = 0;
  int pad_y_ = 0;

  // 从 yaml 读入的阈值，默认值取模型库官方值
  double conf_threshold_;
  double kpt_threshold_;
  int min_valid_kpts_;
  double center_dist_threshold_;

  DecodeStats stats_;  // 仅记录, 不参与任何判定

  // 推理 + 解码，两个公开入口共用
  std::vector<Object> decode(const cv::Mat & image);

  // 在 image 上绘制关键点序号、R 标与类别名（验证关键点顺序用）
  void draw(cv::Mat & image, const std::vector<Object> & objects) const;

  // 转换图像数据: 先转换元素类型, (可选)然后归一化到[0, 1], (可选)然后交换RB通道
  void convert(
    const cv::Mat & input, cv::Mat & output, const bool normalize, const bool exchangeRB) const;

  // 对网络的输入为图片数据的节点进行赋值，实现图片数据输入网络；
  // 逆变换参数写入 scale_ / pad_x_ / pad_y_
  void fill_tensor_data_image(ov::Tensor & input_tensor, const cv::Mat & input_image);

  // 打印模型信息, 这个函数修改自$${OPENVINO_COMMON}/utils/src/args_helper.cpp的同名函数
  void printInputAndOutputsInfo(const ov::Model & network);

  // 将image保存为"../result/$${programName}.jpg"
  void save(const std::string & programName, const cv::Mat & image);
};
}  // namespace auto_buff
#endif

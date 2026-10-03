#ifndef AUTO_BUFF__TRACK_HPP
#define AUTO_BUFF__TRACK_HPP

#include <yaml-cpp/yaml.h>

#include <deque>
#include <optional>

#include "buff_type.hpp"
#include "tools/img_tools.hpp"
#include "yolo11_buff.hpp"
const int LOSE_MAX = 20;  // 丢失的阙值
namespace auto_buff
{
class Buff_Detector
{
public:
  Buff_Detector(const std::string & config);

  std::optional<PowerRune> detect_24(cv::Mat & bgr_img);

  std::optional<PowerRune> detect(cv::Mat & bgr_img);
  // 大符旁路(2026-09-13, 第三轮改为身份锁定): 与 detect() 并存,
  // 只被 --target=big 的测试/生产链路调用。小符链路零改动。
  //
  // 【身份锁定选叶】(取代第二轮的相位槽位法)
  // 相位槽位法依赖 EKF 相位预测, 而 a/ω/φ 收敛要 ~1.6s, 未收敛期相位漂移
  // → 槽位认错叶 → 污染 roll → 相位更漂, 自激循环。身份锁定改用空间连续性:
  //   1. 已锁定: 近邻续锁(与上一帧锁定叶中心最近者), 距离超门或连续 >=2 帧
  //      找不到才判"熄灭"释放锁 —— 单帧掉线不切叶, 宁可空过也不喂错叶;
  //   2. 熄灭/冷启动: 默认取"相对 R 标偏下"的那块(构型: 低头俯仰比抬头平稳,
  //      且云台下压路径短); 高度差在平局带内改按置信度定胜负;
  //   3. 锁定期间不切换 —— 转子转到高处也不换, 换叶的观测代价
  //      (roll 跳 72° 污染转速估计)远大于打高一点板子的瞄准代价。
  // "偏下"用相对融合 R 标的 dy(图像 y 向下为正, dy 大 = 更低), 不用整幅坐标
  // —— 云台在动整幅 y 会漂, 相对量才稳定。
  //
  // 参数 phase_hint_rad 保留但当前不再参与选叶(接口不变, 免牵动调用方)。

  std::optional<PowerRune> detect_big(
    cv::Mat & bgr_img, std::optional<double> phase_hint_rad);

  std::optional<PowerRune> detect_debug(cv::Mat & bgr_img, cv::Point2f v);

  // 透出上一帧的候选统计, 供离线诊断读取。只读, 不影响检测行为。
  const YOLO11_BUFF::DecodeStats & last_decode_stats() const { return MODE_.last_stats(); }
  // 第十五轮: detect_big 每帧回填真实选中项 + 锁定状态机事件。
  // 仅仪表记录, 检测行为零影响。
  void report_big_choice(
    const cv::Point2f & center, float prob, int class_id, int lock_event,
    float lock_dist_px, int cand_count_raw)
  {
    MODE_.mark_chosen(center, prob, class_id);
    MODE_.set_lock_event(lock_event, lock_dist_px, cand_count_raw);
  }
  // 第十五轮补: detect_big 无输出路径清除选中标记(防测试端误读 objects[0])
  void clear_big_choice() { MODE_.clear_chosen(); }


private:
  void handle_lose();

  YOLO11_BUFF MODE_;
  Track_status status_;
  int lose_;  // 丢失的次数
  std::optional<PowerRune> last_powerrune_ = std::nullopt;

  // ---- 大符身份锁定状态(仅 detect_big 使用, 小符路径不碰) ----
  bool big_locked_ = false;                  // 是否有锁定叶
  cv::Point2f big_locked_center_{-1.f, -1.f}; // 锁定叶上一帧中心(像素)
  int big_lock_miss_ = 0;                    // 连续找不到锁定叶的帧数

  // ---- 第十四轮: 释放后幽灵锁(ghost lock) ----
  // 新录像实测 ~97fps, miss>=2 只有 ~21ms: 光线抖动造成的两帧暂时丢检测
  // 会被误判成"熄灭", 释放后立即按"偏下"重捕获 —— 若那一瞬 B 板恰好更低,
  // 就永久切到 B(A 再被检测到也回不去)。这正是"两板都没熄灭却换板"的机制。
  // 修复: 释放锁时保留幽灵位置 + TTL, TTL 内优先在幽灵位置附近找回原叶;
  // 超时才认"真熄灭"走偏下规则。相邻叶心角距 72 度, 转子转 36 度(半个叶位)
  // 用时 >172ms(峰值 2.09 rad/s) 即 >16 帧(97fps), 幽灵窗 8 帧(~83ms)内
  // 原叶必在幽灵位附近, 而 B 板已离开幽灵位至少半个叶位, 不会误锁。
  cv::Point2f big_ghost_center_{-1.f, -1.f}; // 幽灵锁位置(释放时的锁中心)
  int big_ghost_ttl_ = 0;                    // 幽灵剩余帧数, 0=无幽灵
};
}  // namespace auto_buff
#endif  // DETECTOR_HPP

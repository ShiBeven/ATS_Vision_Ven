#include "buff_detector.hpp"

#include "tools/logger.hpp"

namespace auto_buff
{
Buff_Detector::Buff_Detector(const std::string & config) : status_(LOSE), lose_(0), MODE_(config) {}

// 旧的 get_r_center() / handle_img() 已删除：新模型(Rune-v8)直接输出 R 标关键点，
// 不再需要"灰度二值化 + mask + 轮廓打分"那套靠图像处理猜 R 标的启发式。

void Buff_Detector::handle_lose()
{
  lose_++;
  if (lose_ >= LOSE_MAX) {
    status_ = LOSE;
    last_powerrune_ = std::nullopt;
  }
  status_ = TEM_LOSE;
}

std::optional<PowerRune> Buff_Detector::detect_24(cv::Mat & bgr_img)
{
  /// 模型检测

  std::vector<YOLO11_BUFF::Object> results = MODE_.get_multicandidateboxes(bgr_img);

  /// 处理未获得的情况

  if (results.empty()) {
    handle_lose();
    return std::nullopt;
  }

  /// results转扇叶FanBlade

  std::vector<FanBlade> fanblades;
  for (auto & result : results)
    fanblades.emplace_back(FanBlade(result.kpt, result.center, _light));

  /// 生成PowerRune。R 标与类别取置信度最高的候选(NMS 已按置信度降序输出)
  PowerRune powerrune(fanblades, results[0].r, last_powerrune_, results[0].class_id);

  /// handle error
  if (powerrune.is_unsolve()) {
    handle_lose();
    return std::nullopt;
  }

  status_ = TRACK;
  lose_ = 0;
  std::optional<PowerRune> P;
  P.emplace(powerrune);
  last_powerrune_ = P;
  return P;
}

std::optional<PowerRune> Buff_Detector::detect(cv::Mat & bgr_img)
{
  /// 模型检测

  std::vector<YOLO11_BUFF::Object> results = MODE_.get_onecandidatebox(bgr_img);

  /// 处理未获得的情况

  if (results.empty()) {
    handle_lose();
    return std::nullopt;
  }

  /// results转扇叶FanBlade

  std::vector<FanBlade> fanblades;
  auto & result = results[0];
  fanblades.emplace_back(FanBlade(result.kpt, result.center, _light));

  /// 生成PowerRune。R 标由网络直出，不再外推猜测
  PowerRune powerrune(fanblades, result.r, last_powerrune_, result.class_id);

  /// handle error
  if (powerrune.is_unsolve()) {
    handle_lose();
    return std::nullopt;
  }

  status_ = TRACK;
  lose_ = 0;
  std::optional<PowerRune> P;
  P.emplace(powerrune);
  last_powerrune_ = P;
  return P;
}

std::optional<PowerRune> Buff_Detector::detect_debug(cv::Mat & bgr_img, cv::Point2f v)
{
  /// 模型检测

  std::vector<YOLO11_BUFF::Object> results = MODE_.get_multicandidateboxes(bgr_img);

  /// 处理未获得的情况

  if (results.empty()) return std::nullopt;

  /// results转扇叶FanBlade

  std::vector<FanBlade> fanblades_t;
  for (auto & result : results) fanblades_t.emplace_back(FanBlade(result.kpt, result.center, _light));

  /// 筛选fanblade。R 标取置信度最高的候选
  auto r_center = results[0].r;
  std::vector<FanBlade> fanblades;
  for (auto & fanblade : fanblades_t) {
    if (cv::norm((fanblade.center - r_center) - v) < 10 || results.size() == 1) {
      fanblades.emplace_back(fanblade);
      break;
    }
  }
  if (fanblades.empty()) return std::nullopt;
  PowerRune powerrune(fanblades, r_center, std::nullopt, results[0].class_id);

  std::optional<PowerRune> P;
  P.emplace(powerrune);
  return P;
}

}  // namespace auto_buff

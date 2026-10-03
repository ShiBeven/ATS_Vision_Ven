// 打符全链路离线验证入口（检测 → PnP → EKF → Aimer）。
//
// 与 buff_detect_video_test 的分工：那个只跑检测层看关键点，这个跑到 EKF 和瞄准，
// 用来做步骤 13 的 PnP 合理性验证。
//
// 录像旁若有同名 .txt（每行 `t w x y z`，四元数日志）就用它；没有则退化为
// 单位四元数，即把云台系当世界系。此时 R_dis / angle / spd 这些量的内部一致性
// 判据依然成立（判据 1/2/3/4），只有涉及真实云台朝向的判据 6 失去意义。

#include <deque>  // 第十五轮: 切板取证环形缓冲
#include <fmt/core.h>
#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/path.hpp"
#include "tools/plotter.hpp"

// 注意：config 与录像都是位置参数，顺序为 <config> <video>，与 buff_detect_video_test 一致。
// 改动前 config 是 `-c` 命名参数、录像是第 1 个位置参数，
// 于是 `auto_buff_test configs/standard3.yaml 录像.avi` 会把 yaml 当成录像路径。
const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明                    }"
  "{start-index s  | 0                      | 视频起始帧下标                        }"
  "{end-index e    | 0                      | 视频结束帧下标                        }"
  "{target t       | small                  | 观测器类型: small(小符) / big(大符)   }"
  "{fps            | 0                      | 覆盖录像自报帧率, 0=用录像声明的值    }"
  "{no-stage2      |                        | 诊断: 关掉 EKF 第二级(叶心)更新       }"
  "{@config-path   | configs/standard3.yaml | yaml配置文件的路径                    }"
  "{@input-path    | assets/demo/符.avi     | 录像路径, 带或不带 .avi 后缀均可      }";

// 一路累计的统计量。步骤 13 的判据基本都能从这里读出来，
// 这样不必只靠肉眼看 PlotJuggler 曲线，控制台摘要可以直接贴出来对照。
struct RunStats
{
  int frames = 0;
  int detected = 0;
  int solved = 0;
  int lost_after_solved = 0;  // 曾经解出过又变成 unsolve 的次数(含 EKF 发散重置)

  // 原始 PnP 观测
  double dis_sum = 0, dis_sq_sum = 0, dis_min = 1e9, dis_max = -1e9;
  int dis_n = 0;
  double rms_sum = 0, rms_max = -1e9;
  int rms_n = 0;

  // EKF 输出
  double spd_sum = 0, spd_sq_sum = 0, spd_min = 1e9, spd_max = -1e9;
  int spd_n = 0;
  void add_dis(double d)
  {
    dis_sum += d;
    dis_sq_sum += d * d;
    dis_min = std::min(dis_min, d);
    dis_max = std::max(dis_max, d);
    dis_n++;
  }

  void add_spd(double s)
  {
    spd_sum += s;
    spd_sq_sum += s * s;
    spd_min = std::min(spd_min, s);
    spd_max = std::max(spd_max, s);
    spd_n++;
  }


  int angle_step_count = 0;   // |Δangle| 超过半个扇叶间距(36°)的帧数 = 疑似换叶/跳变
  double angle_rate_abs_max = 0;
  int yaw_pitch_flip = 0;     // buff_yaw 与 buff_pitch 同帧成对变号的次数

  std::map<int, int> class_hist;

  // ---- 分离"帧间换叶"与"IPPE 二义性"的三组仪表 ----
  //
  // 两个故障都会让 roll 跳变, 单看 angle 分不开, 所以要分别取证:
  //   换叶     → 候选数 > 1, 且选中叶的像素中心在帧间大幅跳动
  //   IPPE二义 → 候选数恒为 1、中心不跳, 但 yaw/pitch 成对变号
  std::map<int, int> candidate_count_hist;  // 每帧 NMS 后候选数的分布
  int chosen_center_jump = 0;               // 选中叶中心帧间跳动 > 80 px 的帧数
  double chosen_center_jump_max = 0;
  int chosen_class_switch = 0;              // 选中叶的 class_id 在帧间改变的次数

  // roll 跳变按 72° 的整数倍归类: key = round(|Δroll| / 72°), value = 帧数。
  // 换叶产生的跳变会集中在 1、2 这些整数倍上; 二义性/噪声则落在 0 桶里。
  std::map<int, int> roll_step_hist;

  // ---- 缺口 1: 中心跳发生在单候选帧还是多候选帧 ----
  //
  // 107 次中心跳若全落在多候选帧, 那就是 get_onecandidatebox() 在多个候选里
  // 选错了叶, 换个选法即可修。但若单候选帧(该帧 NMS 后只剩 1 个候选, 上层
  // 根本无从选起)也在跳, 说明问题出在 NMS/置信度层 —— 每帧只吐一个、
  // 且吐的是不同的叶。那种情况下任何"改选叶策略"的修法都无效, 必须回去
  // 重新定位。这两种成因的修法完全不同, 所以必须分开计数。
  int center_jump_single_cand = 0;  // 跳变帧中, 当帧候选数 == 1 的
  int center_jump_multi_cand = 0;   // 跳变帧中, 当帧候选数 >= 2 的

  // ---- 缺口 2: 每帧 class_id==0(未激活)的候选数分布 ----
  //
  // 决定"按 class0 过滤"这个修法能否唯一确定目标: 若 class0 恒为 1 片,
  // 过滤完就只剩一个候选, 换叶被根治; 若常有 2 片以上, 过滤只能剔掉已激活的叶,
  // 剩下的仍要靠别的手段选, 只减轻不根治。
  std::map<int, int> class0_count_hist;

  // ---- 故障 3(spd 系统性正偏) 定位用: 原始 roll 观测的表观角速度 ----
  //
  // 现象: spd 被单向推到 70 撞门, 下界恰为初值 60, 节律规整。这有两个方向相反的解释,
  // 必须先量一次才能选修法, 否则一半概率越改越糟:
  //   观测 ~60 deg/s → 观测本身是对的, 正偏纯属滤波器缺陷(buff_target.cpp 的 Q_ 里
  //                    roll/spd 两行全零, 残差无处可去只能被解释成转速变化) → 补 Q_ 方向对
  //   观测 62~65     → 录像里表观转速本就高于 60, spd 是在"正确"跟踪它, 撞门只是因为
  //                    init 强制从 SMALL_W=60 起、发散门开在 ±10 → 补 Q_ 方向反了
  //
  // 两条纪律:
  //   1. 只统计"连续两帧都有检测 且 k==0"的帧对。k>=1 是整叶位跳(重捕获/换叶),
  //      单帧上百度, 混进来直接把均值抬飞。被排除的帧对单独计数, 免得静默丢样本。
  //   2. 同时打 deg/帧。deg/s 依赖录像自报帧率(run() 里 t = frame_count / fps),
  //      deg/帧 不依赖。两个数对照 fps 日志就能查出帧率报错这一种可能:
  //      官方小符 60 deg/s, 50 fps 应为 1.2 deg/帧, 30 fps 应为 2.0 deg/帧。
  //
  // 本仪表只读, 不改任何行为。
  std::vector<double> raw_roll_rates;  // 有符号, rad/s, 存全量是为了取 median
  double raw_roll_rate_sum = 0, raw_roll_rate_sq_sum = 0;
  double raw_roll_step_sum = 0;   // 有符号, rad/帧
  int raw_roll_rate_n = 0;
  int raw_roll_rate_skipped = 0;  // 因 k>=1(整叶位跳)被排除的帧对数

  void add_raw_roll_rate(double signed_d_roll, double dt)
  {
    if (!(dt > 0)) return;
    const double rate = signed_d_roll / dt;
    raw_roll_rates.push_back(rate);
    raw_roll_rate_sum += rate;
    raw_roll_rate_sq_sum += rate * rate;
    raw_roll_step_sum += signed_d_roll;
    raw_roll_rate_n++;
  }

  // ---- 用户判据: 绿/蓝线的转轴端必须钉在检测到的 R 十字上 ----
  //
  // 用户的原话是"不论怎么跳变, 它的一边都不可能离开 R 标"。落到几何上:
  // OBJECT_POINTS 第二组(叶心 700 → 转轴 0)画出来的那条线, 末端就是 buff 系原点,
  // 而 R 标志的中心就是转轴 —— 所以那个端点必须落在洋红十字上。这里把"必须"量化。
  //
  // 三组量各有分工, 不要混读:
  //   r_ratio / r_perp  纯 2D, 不用位姿, 不受共面 PnP 病态影响 → 判 R 该配哪个物体点
  //   pnp_r_err         PnP 位姿把 (0,0,0) 投回来离 R 多远 → 判几何层
  //   ekf_r_err         EKF 状态把 (0,0,0) 投回来离 R 多远 → 判滤波层(这才是画面上的绿线)
  // 两个分开看才能定位: PnP 端就偏 → 几何/点集问题; PnP 准而 EKF 偏 → 滤波器问题。
  std::vector<double> r_ratio;    // R 标沿"上→下"延长线的位置, 以 |上-下| 为单位
  std::vector<double> r_perp;     // R 标到该径向线的横向偏移, 同样归一
  int r_gate_fail = 0;            // 几何门判 R不可信、退回 4 点 PnP 的帧数
  std::vector<double> pre_r_err;  // 蓝线(预测)轴端离 R 十字的距离, px

  std::vector<double> pnp_r_err;  // px
  std::vector<double> ekf_r_err;  // px

  // ---- 第二十轮: 绿框漂移分量分解(大符绿框右下漂移定位) ----
  // 画面现象: 绿框相对检测叶偏右下 60~90px。这个偏移必须先分解到状态分量才能定位:
  //   green_dr  px    绿框叶心点与检测叶心(700mm 点, 由 PnP 投影)的欧氏距离
  //   green_droll  deg 绿框叶心相对检测叶心绕 R 的角度差(有符号, 限 ±180°)
  //   green_dradius px 绿框叶心相对检测叶心的径向长度差(|green|-|obs|, 有符号)
  //   green_dpos  px   绿框叶心与检测叶心在图像 x/y 方向的位置差(有符号)
  //   green_yaw_state  deg  EKF x[4] 与 PnP ypr[0] 的差 —— 叶面法向偏航误差
  //   green_R_axis_err  px  绿框轴端(image_points[6])与检测 R 标的距离 —— 转轴状态误差
  std::vector<double> green_dr;      // px, 有符号
  std::vector<double> green_droll;   // deg, 有符号
  std::vector<double> green_dradius; // px, 有符号
  std::vector<double> green_dx;      // px, 有符号
  std::vector<double> green_dy;      // px, 有符号
  std::vector<double> green_yaw_state;  // deg, 有符号
  std::vector<double> green_R_axis_err; // px
  // ---- 第二十二轮: 最差帧定位(找出大误差发生在哪些帧段) ----
  // med 已收敛但 mean|.|/max 大 → 误差集中在少数帧段。记录每帧元数据,
  // 结尾输出 |d| 最大的帧及其分量, 与切板事件日志按帧号对齐。
  std::vector<int> green_frame;
  std::vector<double> green_time;
  std::vector<double> green_phase;
  // 解算路径: 1=used_r(R标5点+LM), 2=4点回退
  std::vector<int> green_solve_path;
  void add_green_drift(
    double dr_deg, double dradius_px, double dx_px, double dy_px, double yaw_diff_deg,
    double axis_err_px, int frame, double time, double phase, int solve_path)
  {
    green_dr.push_back(std::hypot(dx_px, dy_px));
    green_droll.push_back(dr_deg);
    green_dradius.push_back(dradius_px);
    green_dx.push_back(dx_px);
    green_dy.push_back(dy_px);
    green_yaw_state.push_back(yaw_diff_deg);
    green_R_axis_err.push_back(axis_err_px);
    green_frame.push_back(frame);
    green_time.push_back(time);
    green_phase.push_back(phase);
    green_solve_path.push_back(solve_path);
  }

  // ---- 环带门仪表(2026-09-13, 大符臂灯杂点) ----
  std::vector<double> cand_radius_ratios;  // 候选中心 r/baseline
  std::vector<double> kpt_radius_ratios;    // 四角点 r/baseline
  int ring_cand_reject_total = 0;
  int ring_kpt_reject_total = 0;

  // ---- EKF 那 134 px 的分量分解: 偏差落在哪个状态上 ----
  //
  // 第一轮实测 PnP median 3.0 px / EKF median 133.9 px —— 观测层已经把转轴钉在
  // R 十字上了, 偏差整个产生在滤波层。要修必须先知道是哪个状态偏, 否则又是猜。
  //
  // 134 px / fx 2367 = 56.6 mrad = 3.24°, 所以偏差**主要是横向(方向角)**:
  // 径向距离误差在 9.4 m 处几乎不动像素, 单靠 R_dis 偏不出 134 px 来。
  //
  // 五个量各自否证一种机理:
  //   d_ryaw / d_rpitch  EKF 状态减去同帧 PnP 观测。哪个接近 3.2° 就是哪个没跟上
  //   d_rdis             同上, 米。用来确认偏差确实不在径向
  //   buff_pitch         PnP 解出的符面 pitch(ypr[1])。这个量**状态里根本没有**,
  //                      point_buff2world() 把 pitch 写死成 0 —— 若它显著非零, 则阶段 2
  //                      (叶心观测)的模型有结构性误差, 而残差只能由中心位置吸收
  //   v_ryaw             状态 1。本录像无 IMU 数据, R_gimbal2world 恒为 I, 转轴在世界系里
  //                      静止, 所以它应当恒 ~0。不为 0 说明残差被灌进速度项并积分成漂移
  std::vector<double> d_ryaw;      // deg, 有符号
  std::vector<double> d_rpitch;    // deg, 有符号
  std::vector<double> d_rdis;      // m,   有符号
  std::vector<double> buff_pitch;  // deg, 有符号
  std::vector<double> v_ryaw;      // deg/s, 有符号

  // EKF 协方差矩阵的对角, 只留最后一帧的值(判据是"收到多小", 不需要时间序列)。
  std::vector<double> p_diag_last;

  // ---- 中心跳的性质区分: 合法交接 vs 异常跳 ----
  //
  // 用户指出的一条领域事实: 目标叶被击打后会熄灭, 另一片随之亮起, 此时选中叶
  // 换到别的扇臂**是正确行为**, 不是代码缺陷。所以 41 次中心跳不能整体当故障读。
  //
  // 判别信号: 已激活候选数(class_id != 0 的个数)在这一帧或紧邻的前几帧增加了。
  // 一片叶被打亮 → 它从 class0 变成 class1 → get_onecandidatebox() 只在 class0 里选,
  // 目标必然换到另一条扇臂。逐窗候选直方图正好显示激活是渐进发生的
  // (200 帧 1:83 → 800 帧 1:379 2:167 3:121), 与这个机理自洽。
  //
  // 容差取 3 帧: 亮起与熄灭在检测层可能差一两帧才反映出来。
  int center_jump_handover = 0;
  int center_jump_abnormal = 0;
  int activation_events = 0;  // 已激活候选数增加的次数(5 片叶, 正常不超过 4~5 次)

};



// ---- 第十五轮: 切板取证基础设施 ----
// 环形缓冲记最近 10 帧的选叶全景; 切板(选中中心跳>80px 或 EKF 解后丢失)时
// 打印缓冲 + 后续 5 帧。平时零输出。
struct FrameTrace
{
  int frame = -1;
  double t = 0;
  cv::Point2f chosen{-1.f, -1.f};
  int chosen_class = -1;
  float chosen_prob = 0.f;
  int lock_event = 0;   // 0无 1续锁 2丢1 3释放 4幽灵找回 5幽灵未中 6偏下 7冷启动
  float lock_dist = -1.f;
  int cand_count = 0;   // 环带门后候选数
  int cand_count_raw = 0;  // 环带门前原始候选数
  std::vector<std::pair<cv::Point2f, float>> cands;  // 候选(中心, 置信度)
  bool cands_truncated = false;
};

void print_trace(const FrameTrace & ft)
{
  std::string ev;
  switch (ft.lock_event) {
    case 0: ev = "NONE"; break;
    case 1: ev = "KEEP"; break;
    case 2: ev = "MISS1"; break;
    case 3: ev = "RELEASE"; break;
    case 4: ev = "GHOST_HIT"; break;
    case 5: ev = "GHOST_MISS"; break;
    case 6: ev = "LOWER"; break;
    case 7: ev = "COLDSTART"; break;
    default: ev = "?";
  }
  std::string cands;
  for (const auto & c : ft.cands) {
    cands += fmt::format("({:.0f},{:.0f})p{:.2f} ", c.first.x, c.first.y, c.second);
  }
  if (ft.cands_truncated) cands += "...";
  if (ft.chosen.x < 0) {
    tools::logger()->info(
      "  帧{} t={:.3f} | 无选中 | {} | 候选 {}/{}", ft.frame, ft.t, ev, ft.cand_count,
      ft.cand_count_raw);
  } else {
    tools::logger()->info(
      "  帧{} t={:.3f} | 选中({:.0f},{:.0f}) cls{} p{:.2f} | {} d{:.1f}px | 候选 {}",
      ft.frame, ft.t, ft.chosen.x, ft.chosen.y, ft.chosen_class, ft.chosen_prob, ev,
      ft.lock_dist, cands.empty() ? "0" : cands);
  }
}

template <typename TargetT>
void run(
  cv::VideoCapture & video, const std::string & video_path, std::ifstream & text, bool has_quaternion,
  double fps, const std::string & config_path, int start_index, int end_index, tools::Plotter & plotter,
  tools::Exiter & exiter, bool no_stage2, bool big = false)
{
  auto_buff::Buff_Detector detector(config_path);
  auto_buff::Solver solver(config_path);
  TargetT target;
  auto_buff::Aimer aimer(config_path);

  // 大符旁路开关: small 走原 detect(), 行为零改动; big 走 detect_big(相位槽位选叶)。
  // 相位提示来自上一帧 EKF 状态 x[5](转子相位), 未解算时 nullopt → 检测层退化最高分。
  const bool use_big_detect = big;

  // 诊断开关, 默认 false → 生产行为逐位不变。开了之后 EKF 只做第一级(R 标)更新。
  target.diag_skip_stage2 = no_stage2;

  RunStats stats;
  cv::Mat img;
  const auto t0 = std::chrono::steady_clock::now();

  video.set(cv::CAP_PROP_POS_FRAMES, start_index);
  if (has_quaternion) {
    for (int i = 0; i < start_index; i++) {
      double t, w, x, y, z;
      text >> t >> w >> x >> y >> z;
    }
  }

  bool prev_solved = false;
  bool has_prev_angle = false;
  double prev_angle = 0, prev_t = 0;
  double prev_buff_yaw = 0, prev_buff_pitch = 0;
  bool has_prev_ypr = false;

  // ---- 第十五轮: 切板取证状态 ----
  // 环形缓冲最近 10 帧; 切板(选中中心跳>80px 或 EKF 解后丢失)触发打印
  // 缓冲全部 + 后续 5 帧。平时零输出。
  std::deque<FrameTrace> trace_buf;
  cv::Point2f switch_prev_center{-1.f, -1.f};
  bool switch_prev_valid = false;
  int switch_pending = 0;   // >0 = 正在打印事件后续帧
  bool prev_solved_trace = false;

  // 换叶取证用: 上一帧被选中那片叶的像素中心、class_id、原始 roll
  cv::Point2f prev_chosen_center{-1.f, -1.f};
  int prev_chosen_class = -1;
  double prev_raw_roll = 0;
  double prev_raw_roll_t = 0;  // 上一次有检测那帧的时间戳, 供表观角速度取 dt
  bool has_prev_chosen = false;
  bool has_prev_raw_roll = false;


  // ---- 重复帧检测(第八轮): 剔除录像 VFR→CFR 填充产生的"复制上帧"假帧 ----
  // 背景: 符.avi 实测 29% 的帧是 H.264 skip 帧(stsz <1KB, 最小 30B),
  // 时间戳声称均匀 30fps 但有效内容仅 ~21fps。这些帧上 raw_roll 与上帧
  // 几乎不变 → 差分出 ≈0°/s 假观测; 下一个内容帧跨过重复帧 → 差分出
  // 2~3 倍真实值的假速度。帧356~360 roll 累加 = -120°/s(真实峰值),
  // 证明图像内容正常、只是时间轴有假帧。
  // 处理: 图像差分判重后, 重复帧的时间戳钉在上一内容帧(即 dt=0)。
  // EKF 对 dt=0 的 predict 是零步长, update 观测与上帧相同, 无害;
  // 差分仪表靠 dt>1e-6 保护自然跳过, 不需要额外断开差分链。
  // 只在离线测试做——生产链路接相机实时流, 无重复帧, 逻辑不进 tasks/。
  cv::Mat prev_content_frame;  // 上一"内容帧"的灰度缩略图(差分基准)
  bool has_prev_content = false;
  long dup_frame_skipped = 0;   // 被判重复的帧数
  long content_frame_cnt = 0;   // 内容帧数
  double dup_diff_sum = 0.0;    // 差分均值累计(供阈值标定)
  long dup_diff_cnt = 0;
  double last_content_t = 0.0;   // 上一内容帧时间戳(重复帧钉回用)
  double dup_threshold = 1.0;         // 判重阈值: run() 开头预扫描自动标定(见下)

  // 合法交接判别用: 上一帧的已激活候选数, 以及距最近一次"激活数增加"过了几帧。
  // 初值给一个大数, 免得开头几帧的中心跳被误判成交接。
  int prev_activated = 0;
  bool has_prev_activated = false;
  int frames_since_activation = 1000;

  // ---- 第十轮: 判重阈值预扫描标定 ----
  // 第九轮实测(阈 1.0): 判重 271(46.6%), 但仍有 -15/-17°/s 的近重复帧漏判 ——
  // 固定阈值落在双峰之间的灰区。健康判据: 差分值呈双峰(近 0 的 skip 峰 vs
  // 大幅运动的内容峰), 谷底才是正确阈值。这里全片预扫一遍差分序列, 取直方图
  // 的最大间隔中点为阈值; 若无双峰(健康录像), 全片差分都大, 阈值落在最大
  // 差分之上 → 判重恒 0, 行为与无检测一致。
  {
    // 第十三轮: 预扫描改用独立 VideoCapture 句柄。
    // 根因: 旧代码在主 video 上读到 EOF 后用 CAP_PROP_POS_FRAMES seek 回退,
    // 对 OpenCV 流式写出的 MJPG AVI(RIFF size=0xFFFFFFFF, 无 idx1 索引)seek 失效
    // —— 主循环第一帧就空读退出, 秒退无任何仪表输出(2026-09-13 23:51 实测)。
    // 独立句柄对主 video 的位置零影响, 任何容器格式都成立。
    cv::VideoCapture prescan(video_path);
    if (!prescan.isOpened()) {
      tools::logger()->warn("预扫描打不开录像(判重阈值按禁用处理): {}", video_path);
    }
    cv::Mat f0, f1;
    std::vector<double> diffs;
    while (prescan.read(f1)) {
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
    prescan.release();
    if (diffs.size() >= 10) {
      std::vector<double> sorted_diffs = diffs;
      std::sort(sorted_diffs.begin(), sorted_diffs.end());
      // 双峰性检验: 只有"近零簇"(差分 < 0.5, 即解码噪声级)占比 ≥ 10% 才认为
      // 录像含重复帧。健康录像(全内容帧)相邻差分都在 3 以上, 近零簇为 0,
      // 此时禁用判重 —— 否则谷底搜索会把阈值放到噪声间隔上, 全片误判。
      size_t near_zero = 0;
      for (double d : sorted_diffs)
        if (d < 0.5) near_zero++;
      if (static_cast<double>(near_zero) / static_cast<double>(sorted_diffs.size()) < 0.10) {
        dup_threshold = std::numeric_limits<double>::max();  // 禁用
        tools::logger()->info(
          "[判重阈值] 预扫描 {} 帧: 近零帧占比 {:.1f}% < 10%, 判定健康录像, 重复帧检测禁用",
          diffs.size(), 100.0 * static_cast<double>(near_zero) / static_cast<double>(sorted_diffs.size()));
      } else {
        // 双峰谷底: 低簇上界与高簇下界之间的最大间隔中点。
        // 低簇 = 近零簇外延(到谷底), 只在 < 10 的低区找, 排除场景突变野值。
        size_t best_i = 0;
        double best_gap = 0.0;
        for (size_t i = 0; i + 1 < sorted_diffs.size(); i++) {
          const double gap = sorted_diffs[i + 1] - sorted_diffs[i];
          if (sorted_diffs[i] < 10.0 && gap > best_gap) {
            best_gap = gap;
            best_i = i;
          }
        }
        dup_threshold = (sorted_diffs[best_i] + sorted_diffs[best_i + 1]) / 2.0;
        tools::logger()->info(
          "[判重阈值] 预扫描 {} 帧, 阈值 {:.2f} (谷底两侧 {:.2f}/{:.2f}), 近零帧 {}/{}",
          diffs.size(), dup_threshold, sorted_diffs[best_i], sorted_diffs[best_i + 1],
          near_zero, sorted_diffs.size());
      }
    }
  }

  for (int frame_count = start_index; !exiter.exit(); frame_count++) {
    if (end_index > 0 && frame_count > end_index) break;

    video.read(img);
    if (img.empty()) break;
    stats.frames++;

    // ---- 重复帧判定(第八轮): 与上一内容帧做降采样灰度差分 ----
    // 阈值 1.0(0~255 灰度均值): 大符相邻内容帧叶片移动 3~5°, 边缘扫过
    // 全画面相当比例像素, 均值差远大于 1; skip 帧的解码噪声 <0.5。
    // 先用仪表统计验证阈值, 不拍脑袋定案(见循环尾部累计输出)。
    bool is_dup_frame = false;
    {
      cv::Mat gray_small;
      cv::cvtColor(img, gray_small, cv::COLOR_BGR2GRAY);
      cv::resize(gray_small, gray_small, {}, 0.25, 0.25);
      if (has_prev_content) {
        cv::Mat diff;
        cv::absdiff(gray_small, prev_content_frame, diff);
        const double mean_diff = cv::mean(diff)[0];
        dup_diff_sum += mean_diff;
        dup_diff_cnt++;
        is_dup_frame = (mean_diff < dup_threshold);
      }
      if (!is_dup_frame) {
        prev_content_frame = gray_small.clone();
        has_prev_content = true;
        content_frame_cnt++;
      } else {
        dup_frame_skipped++;
      }
    }

    // 时间戳: 有日志用日志的 t, 没有就按帧率推算。EKF 的 dt 靠这个, 不能省。
    // 第八轮: 重复帧的 t 钉在上一内容帧时刻 —— 该帧图像是复制的, 它在
    // 物理时间轴上不存在, 让 EKF/差分仪表看到 dt=0 而不是假 dt=1/fps。
    double t = frame_count / fps;
    Eigen::Quaterniond q(1, 0, 0, 0);
    if (has_quaternion) {
      double tt, w, x, y, z;
      if (text >> tt >> w >> x >> y >> z) {
        t = tt;
        q = Eigen::Quaterniond(w, x, y, z);
      } else {
        // 第十二轮: 录像帧数 > 日志行数时终止回放, 而不是回退 frame_count/fps。
        // 新录像(符/MJPG) 解码 1368 帧 vs 日志 1362 行: 若继续跑, 第 1363 帧的 t
        // 从 13.97s 跳回 45.4s(容器 30fps 推算), dt≈31s 的假时间戳会以巨大假速度
        // 尖峰污染 EKF 与 RANSAC。录像与日志必须同步, 耗尽即停。
        tools::logger()->warn(
          "四元数日志在第 {} 帧耗尽, 录像帧数与日志行数不同步, 提前终止回放", frame_count);
        break;
      }
    }
    // 第八轮: 重复帧时间戳钉回上一内容帧。仅无四元数日志时安全(t 纯推算);
    // 有四元数时日志的 t 与云台姿态绑定, 不能改, 此时只靠差分链断开兜底。
    if (is_dup_frame && !has_quaternion) {
      t = last_content_t;
    } else {
      last_content_t = t;
    }
    auto timestamp = t0 + std::chrono::microseconds(int(t * 1e6));

    /// 核心链路

    solver.set_R_gimbal2world(q);

    // 大符: 把 EKF 上一帧转子相位喂给检测层做槽位归位; 小符: 原 detect() 不动。
    std::optional<auto_buff::PowerRune> power_runes;
    if (use_big_detect) {
      std::optional<double> phase_hint;
      if (!target.is_unsolve() && target.ekf_x().size() > 5) phase_hint = target.ekf_x()[5];
      power_runes = detector.detect_big(img, phase_hint);
    } else {
      power_runes = detector.detect(img);
    }

    solver.solve(power_runes);

    target.get_target(power_runes, timestamp);

    auto target_copy = target;

    auto command = aimer.aim(target_copy, timestamp, 22, false);

    // -------------- 调试输出 --------------

    nlohmann::json data;

    if (power_runes.has_value()) {
      stats.detected++;
      auto & p = power_runes.value();  // 非 const: 下面要调 target(), 那是非 const 方法
      data["buff_R_yaw"] = p.ypd_in_world[0];
      data["buff_R_pitch"] = p.ypd_in_world[1];
      data["buff_R_dis"] = p.ypd_in_world[2];
      data["buff_yaw"] = p.ypr_in_world[0] * 57.3;
      data["buff_pitch"] = p.ypr_in_world[1] * 57.3;
      data["buff_roll"] = p.blade_phase * 57.3;  // 第七轮: 几何相位, 与 EKF 观测同源
      data["buff_class_id"] = p.class_id;
      stats.class_hist[p.class_id]++;
      stats.add_dis(p.ypd_in_world[2]);

      // PnP 残差: 4 个观测角点 vs 用本帧位姿把 OBJECT_POINTS 前 4 点投影回来的位置
      const double rms = solver.reprojection_error(p.target().points);
      if (rms >= 0) {
        data["pnp_rms_px"] = rms;
        stats.rms_sum += rms;
        stats.rms_max = std::max(stats.rms_max, rms);
        stats.rms_n++;
      }

      // 判据 5: buff_yaw / buff_pitch 成对变号 = IPPE 法线二义性
      const double yaw_now = p.ypr_in_world[0], pitch_now = p.ypr_in_world[1];
      if (
        has_prev_ypr && yaw_now * prev_buff_yaw < 0 && pitch_now * prev_buff_pitch < 0 &&
        std::abs(yaw_now - prev_buff_yaw) > 10 / 57.3)
        stats.yaw_pitch_flip++;
      prev_buff_yaw = yaw_now;
      prev_buff_pitch = pitch_now;
      has_prev_ypr = true;

      // ---- R 标判据取证 ----
      //
      // 纯 2D, 不用位姿: 与 buff_solver.cpp 里那道几何门同一套算法(那里用来决定
      // R 要不要进 PnP, 这里只记数)。刻意不依赖位姿 —— 4 个共面角点只张 254 mm,
      // 由它外推出的转轴位置本身就是要检验的对象, 不能拿它当尺子量 R。
      const auto & kp = p.target().points;
      if (kp.size() >= 4) {
        const cv::Point2f ax = kp[2] - kp[0];  // 上 → 下, 指向转轴
        const double base = cv::norm(ax);
        if (base > 5.0) {
          const cv::Point2f u = ax / static_cast<float>(base);
          const cv::Point2f w = p.r_center - kp[2];  // 下 → R
          stats.r_ratio.push_back((w.x * u.x + w.y * u.y) / base);
          stats.r_perp.push_back(std::abs(w.x * u.y - w.y * u.x) / base);
        }
      }
      if (!solver.last_solve_used_r()) stats.r_gate_fail++;

      // PnP 位姿下的转轴端离 R 十字多远
      const cv::Point2f r_pnp = solver.point_buff2pixel(cv::Point3f(0, 0, 0));
      stats.pnp_r_err.push_back(cv::norm(r_pnp - p.r_center));

      // ---- 换叶取证 ----
      // 用检测层的候选统计, 而不是 EKF 之后的量: EKF 会把跳变吸收掉,
      // 在它之后看不出候选到底换没换。
      const auto & ds = detector.last_decode_stats();
      stats.candidate_count_hist[ds.candidate_count]++;

      // ---- 环带门仪表采集 ----
      for (float v : ds.cand_radius_ratios) stats.cand_radius_ratios.push_back(v);
      for (float v : ds.kpt_radius_ratios) stats.kpt_radius_ratios.push_back(v);
      stats.ring_cand_reject_total += ds.ring_gate_cand_reject;
      stats.ring_kpt_reject_total += ds.ring_gate_kpt_reject;

      // 已激活候选数(class_id != 0)。它增加 = 有叶刚被打亮 → 目标必然换扇臂,
      // 那种中心跳是合法交接, 不是缺陷。这里维护一个"最近几帧是否发生过激活"的计数器。
      const int activated_now = static_cast<int>(std::count_if(
        ds.class_ids.begin(), ds.class_ids.end(), [](int id) { return id != 0; }));
      if (has_prev_activated && activated_now > prev_activated) {
        stats.activation_events++;
        frames_since_activation = 0;
      } else if (frames_since_activation < 1000) {
        frames_since_activation++;
      }
      prev_activated = activated_now;
      has_prev_activated = true;

      // 缺口 2: 数当帧有几个 class0(未激活)候选。class_ids 是 NMS 后的全部候选,
      // 与 candidate_count 同源, 所以这里不必再判空。
      stats.class0_count_hist[static_cast<int>(
        std::count(ds.class_ids.begin(), ds.class_ids.end(), 0))]++;



      // roll 跳变按 72° 整数倍归类。用的是 PnP 直出的原始 roll,
      // 必须在 buff_target 把它拉到最近 72° 倍数之前取。
      // 注意只在"连续两帧都有检测"时统计: 中间断过帧, 真实旋转本身就会累积角度,
      // 那种跳变不是换叶造成的, 计进来会虚高。
      // 第七轮: raw_roll 改用几何相位 blade_phase, 与 BigTarget EKF 观测同源。
      // 原 ypr[2] 欧拉分解在 IPPE 双解翻转时会把 yaw/pitch 跳变泄漏进 roll,
      // 造成隔帧 -132~-312 deg/s 的物理不可能差分速度(2026-09-13 18:32 日志)。
      const double raw_roll = p.blade_phase;
      if (has_prev_raw_roll) {
        const double signed_d_roll = tools::limit_rad(raw_roll - prev_raw_roll);
        const double d_roll = std::abs(signed_d_roll);
        const int k = static_cast<int>(std::lround(d_roll / (2 * CV_PI / 5)));
        stats.roll_step_hist[k]++;

        // 故障 3 定位: 只有 k==0 的帧对是"同一叶位内的正常旋转", 拿它量表观转速;
        // k>=1 是整叶位跳, 排除并单独计数。dt 用同一对帧的时间差, 不复用 EKF 那套 prev_t
        // (那个只在 solved 分支里推进, 与本统计的连续性条件不同)。
        if (k == 0)
          stats.add_raw_roll_rate(signed_d_roll, t - prev_raw_roll_t);
        else
          stats.raw_roll_rate_skipped++;
      }

    }

    // 本帧没检测 → 断开连续性, 下一帧不与"上一次有检测的帧"作差
    // 第八轮注: 重复帧**不**断开。时间戳已钉回内容帧时刻, 差分块内
    // dt=0 → 所有差分消费端(dt>1e-6 保护)自动失效, 不会输出假速度;
    // 而基线仍会以"钉回的 t"刷新(≈内容帧时刻), 下一个内容帧跨重复帧
    // 作差, dt 正好等于真实内容间隔。若在此断开反而销毁基线,
    // 让重复帧之后的内容帧也丢一次有效观测。
    if (!power_runes.has_value()) {
      has_prev_chosen = false;
      has_prev_raw_roll = false;
      has_prev_ypr = false;
      // 激活数不清: 空窗期间叶片不会变回未激活, 断开只会让重捕获后的第一次
      // 比较凭空产生一个"激活事件"。保留 prev_activated 更贴近物理。
    }
    const bool solved_now = !target.is_unsolve();
    if (solved_now) stats.solved++;
    if (prev_solved && !solved_now) stats.lost_after_solved++;
    prev_solved = solved_now;
      // ---- 第十五轮: 切板取证日志(替换第四~十四轮全部转速仪表) ----
      // 设计: 环形缓冲记最近 10 帧全景(帧号/选中中心/锁定事件/候选列表),
      // 检测到"选中叶跳 >80px"(= 换板)时, 把缓冲连同后续 5 帧一起打印。
      // 平时零输出。数据源: detect_big 回填的真实选中项(修复了旧仪表记录
      // objects[0] 的盲区 —— 之前换叶取证量的根本不是实际选中的那片叶)。
      {
        const auto & ds_now = detector.last_decode_stats();
        const bool has_choice = (ds_now.chosen_center.x >= 0);

        // 每帧先记入环形缓冲
        if (has_choice) {
          FrameTrace ft;
          ft.frame = frame_count;
          ft.t = t;
          ft.chosen = ds_now.chosen_center;
          ft.chosen_class = ds_now.chosen_class_id;
          ft.chosen_prob = ds_now.chosen_prob;
          ft.lock_event = ds_now.lock_event;
          ft.lock_dist = ds_now.lock_dist_px;
          ft.cand_count = ds_now.candidate_count;
          ft.cand_count_raw = ds_now.cand_count_raw;
          for (size_t ci = 0; ci < ds_now.cand_centers.size() && ci < 6; ci++) {
            ft.cands.emplace_back(ds_now.cand_centers[ci], ds_now.cand_probs[ci]);
          }
          if (ds_now.cand_centers.size() > 6) ft.cands_truncated = true;
          trace_buf.push_back(std::move(ft));
          if (trace_buf.size() > 10) trace_buf.pop_front();  // 环形缓冲: 只留最近 10 帧
        }

        // 无检测帧也要留痕(候选空 → 锁定 MISS 的关键现场), 用占位记录
        if (!has_choice) {
          FrameTrace ft;
          ft.frame = frame_count;
          ft.t = t;
          ft.chosen = {-1.f, -1.f};
          ft.cand_count = 0;
          ft.cand_count_raw = ds_now.cand_count_raw;
          ft.lock_event = ds_now.lock_event;
          trace_buf.push_back(std::move(ft));
          if (trace_buf.size() > 10) trace_buf.pop_front();  // 环形缓冲: 只留最近 10 帧
        }

        // 切板判定: 选中中心跳 >80px(与 detect_big 续锁门同阈值)
        bool switched = false;
        if (has_choice && switch_prev_valid) {
          const double jump = cv::norm(ds_now.chosen_center - switch_prev_center);
          if (jump > 80.0) switched = true;
        }
        if (has_choice) {
          switch_prev_center = ds_now.chosen_center;
          switch_prev_valid = true;
        }
        // EKF 解后丢失也触发(可能是锁定释放造成的)
        if (prev_solved_trace && !solved_now) switched = true;

        if (switched) {
          stats.chosen_center_jump++;
          switch_pending = 6;  // 事件帧 + 后续 5 帧继续打印
          if (!trace_buf.empty()) {
            tools::logger()->warn(
              "======== 切板事件 @ 帧{} (t={:.2f}s) 上下文如下 ========", frame_count, t);
            for (const auto & ft : trace_buf) print_trace(ft);
          }
        } else if (switch_pending > 0) {
          // 事件后的后续帧
          if (!trace_buf.empty()) print_trace(trace_buf.back());
          switch_pending--;
          if (switch_pending == 0)
            tools::logger()->info("======== 切板上下文结束 ========");
        }

        prev_solved_trace = solved_now;
      }


    // 第十八轮 keep-alive 修复: solved_now 只表示 EKF 在输出(含无观测的纯 predict 帧),
    // 不保证本帧有检测。旧代码在这里无条件 value() —— keep-alive 生效后第一个
    // 无检测帧(solved=true + nullopt)即抛 bad_optional_access 崩溃。
    // EKF 侧的绿框/蓝框/数据在无检测帧依然有效, 只有依赖 p(检测观测)的部分跳过。
    if (solved_now && power_runes.has_value()) {
      auto & p = power_runes.value();

      // 显示
      for (int i = 0; i < 4; i++) tools::draw_point(img, p.target().points[i]);
      tools::draw_point(img, p.target().center, {0, 0, 255}, 3);
      tools::draw_point(img, p.r_center, {0, 0, 255}, 3);

      // ---- 叠加层的分层约定(为了让"线端有没有连在 R 十字上"能定位到具体哪一层) ----
      //
      //   洋红十字(yolo11_buff::draw)  检测到的 R 标, 是所有判断的基准
      //   白色线                        纯 PnP 位姿的 叶心→转轴 轴线  → 几何层
      //   绿色线                        当前帧 EKF 状态的同一条轴线    → 滤波层
      //   蓝色线                        预测(超前)位置的同一条轴线
      //
      // 三条线的末端都应当落在洋红十字上。哪一条偏了就是哪一层的问题:
      // 白线就偏 → 点集/几何; 白线准而绿线偏 → EKF; 绿线准而蓝线乱 → 预测/spd。
      //
      // 轴线一律画成 2 点直线(叶心 700 → 转轴 0), 不再用 OBJECT_POINTS[4..6] 三点
      // drawContours。原来那样画会经过中间点 220 再折回, 端点是哪个看不出来,
      // 而"端点在不在十字上"正是要看的东西。

      // 白色 = 纯 PnP 位姿, 不经过 EKF
      const cv::Point2f pnp_blade = solver.point_buff2pixel(cv::Point3f(0, 0, 700e-3));
      const cv::Point2f pnp_axis = solver.point_buff2pixel(cv::Point3f(0, 0, 0));
      cv::line(img, pnp_blade, pnp_axis, {255, 255, 255}, 1, cv::LINE_AA);
      cv::circle(img, pnp_axis, 6, {255, 255, 255}, 1, cv::LINE_AA);

      // 绿色 = 当前帧 EKF 状态重投影, 应当贴合符叶
      // R_len/C_len 在这里生效: 径向缩放 + 圆周偏移, 轴端仍钉在 R 标上,
      // 与瞄准点(buff_aimer.cpp 的 aim_point_in_world)是同一套参数。
      auto Rxyz_in_world_now = target.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.0));
      auto image_points = solver.reproject_buff(
        Rxyz_in_world_now, target.ekf_x()[4], target.ekf_x()[5], aimer.R_len(), aimer.C_len());
      tools::draw_points(
        img, std::vector<cv::Point2f>(image_points.begin(), image_points.begin() + 4), {0, 255, 0});
      cv::line(img, image_points[4], image_points[6], {0, 255, 0}, 1, cv::LINE_AA);
      cv::circle(img, image_points[6], 6, {0, 255, 0}, 1, cv::LINE_AA);
      stats.ekf_r_err.push_back(cv::norm(image_points[6] - p.r_center));
      const cv::Point2f green_blade = image_points[4];  // 绿框叶心(700mm 点)
// ---- 第二十轮: 绿框漂移分量分解仪表(生产链路零改动, 只加测量) ----
// green_blade = EKF 状态重投影的叶心; pnp_blade = 纯 PnP 位姿的叶心(700mm 点)。
// 两者都是 700mm 点且都经过同一套投影, 差值就是 EKF 状态(x[4]yaw / x[5]roll /
// x[0..3]R 轴)与几何层(PnP)的偏差, 与 R_len/C_len 无关(两者同乘同组参数)。
// 把欧氏距离分解成绕 R 角度差 + 径向长度差, 能区分:
//   角度差大 -> 相位(roll)状态偏, 框沿圆周漂
//   径向差大 -> 转轴(R_yaw/R_pitch/R_dis)或 yaw 状态偏, 框沿径向漂
{
  const cv::Point2f R_obs = p.r_center;
  const double dx = green_blade.x - pnp_blade.x;
  const double dy = green_blade.y - pnp_blade.y;
  double ang_obs = std::atan2(pnp_blade.y - R_obs.y, pnp_blade.x - R_obs.x) * 57.2957795;
  double ang_grn = std::atan2(green_blade.y - R_obs.y, green_blade.x - R_obs.x) * 57.2957795;
  double d_roll = ang_grn - ang_obs;
  while (d_roll > 180.0) d_roll -= 360.0;
  while (d_roll <= -180.0) d_roll += 360.0;
  const double r_green = std::hypot(green_blade.x - R_obs.x, green_blade.y - R_obs.y);
  const double r_obs = std::hypot(pnp_blade.x - R_obs.x, pnp_blade.y - R_obs.y);
  const double d_radius = r_green - r_obs;
  const double axis_err = cv::norm(image_points[6] - R_obs);
  const double yaw_diff = tools::limit_rad(target.ekf_x()[4] - p.ypr_in_world[0]) * 57.2957795;
  stats.add_green_drift(d_roll, d_radius, dx, dy, yaw_diff, axis_err, frame_count, t, target.ekf_x()[5] * 57.3, solver.last_solve_used_r() ? 1 : 2);
  data["green_dx"] = dx;
  data["green_dy"] = dy;
  data["green_droll"] = d_roll;
  data["green_dradius"] = d_radius;
  data["green_yaw_diff"] = yaw_diff;
}

      // 蓝色 = 预测(超前)位置, 应当沿旋转方向领先绿色
      auto Rxyz_in_world_pre = target_copy.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.0));
      image_points = solver.reproject_buff(
        Rxyz_in_world_pre, target_copy.ekf_x()[4], target_copy.ekf_x()[5], aimer.R_len(),
        aimer.C_len());
      tools::draw_points(
        img, std::vector<cv::Point2f>(image_points.begin(), image_points.begin() + 4), {255, 0, 0});
      cv::line(img, image_points[4], image_points[6], {255, 0, 0}, 1, cv::LINE_AA);
      stats.pre_r_err.push_back(cv::norm(image_points[6] - p.r_center));

      // 观测器内部数据
      Eigen::VectorXd x = target.ekf_x();
      data["R_yaw"] = x[0];
      data["R_V_yaw"] = x[1];
      data["R_pitch"] = x[2];
      data["R_dis"] = x[3];
      data["yaw"] = x[4] * 57.3;
      data["angle"] = x[5] * 57.3;
      data["spd"] = x[6] * 57.3;
      stats.add_spd(x[6]);

      // ---- EKF 偏差的分量分解 (判读见 RunStats 里 d_ryaw 那段注释) ----
      // 与 ekf_r_err 同一批帧, 所以两组数字能对上 —— 但要拿 **med|.|** 去对, 不是 median:
      // median 是有符号的, 量的是偏置; 折成像素该和 ekf_r_err 比的是 med|.| × 41.3 px/°。
      // (这条原先写成"median 接近就闭合了", 照此判读过一次并得出了错的结论。)
      stats.d_ryaw.push_back(tools::limit_rad(x[0] - p.ypd_in_world[0]) * 57.3);
      stats.d_rpitch.push_back(tools::limit_rad(x[2] - p.ypd_in_world[1]) * 57.3);
      stats.d_rdis.push_back(x[3] - p.ypd_in_world[2]);
      stats.buff_pitch.push_back(p.ypr_in_world[1] * 57.3);
      stats.v_ryaw.push_back(x[1] * 57.3);

      // 每帧覆盖, 跑完留下的就是末值
      const Eigen::VectorXd p_diag = target.ekf_P_diag();
      stats.p_diag_last.assign(p_diag.data(), p_diag.data() + p_diag.size());

      // 判据 1: angle 连续性。注意 x[5] 未被 limit_rad 到单圈之外, 这里只看相邻帧增量。
      if (has_prev_angle && t > prev_t) {
        const double dangle = tools::limit_rad(x[5] - prev_angle);
        if (std::abs(dangle) > 36 / 57.3) stats.angle_step_count++;
        const double rate = dangle / (t - prev_t);
        stats.angle_rate_abs_max = std::max(stats.angle_rate_abs_max, std::abs(rate));
        data["angle_rate"] = rate * 57.3;
      }
      prev_angle = x[5];
      prev_t = t;
      has_prev_angle = true;

      if (x.size() >= 10) {
        data["spd"] = x[6];
        data["a"] = x[7];
        data["w"] = x[8];
        data["fi"] = x[9];
        data["spd0"] = target.spd;
      }
    } else {
      has_prev_angle = false;
    }

    // 云台响应情况
    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
    data["gimbal_yaw"] = ypr[0] * 57.3;
    data["gimbal_pitch"] = -ypr[1] * 57.3;

    if (command.control) {
      data["cmd_yaw"] = command.yaw * 57.3;
      data["cmd_pitch"] = command.pitch * 57.3;
    }

    plotter.plot(data);


    cv::Mat show;
    cv::resize(img, show, {}, 0.6, 0.6);
    cv::imshow("result", show);

    int key = cv::waitKey(1);
    if (key == 'q') break;
    while (key == ' ') {  // 空格暂停, 再按空格继续, q 退出
      int k = cv::waitKey(30);
      if (k == ' ' || k == 'q') {
        key = k;
        break;
      }
    }
    if (key == 'q') break;
  }


  // ---- 第二十轮: 绿框漂移分量分解摘要(结尾一次性输出, 只读) ----
  if (!stats.green_dr.empty()) {
    auto median_of = [](std::vector<double> v) {
      if (v.empty()) return 0.0;
      std::sort(v.begin(), v.end());
      return v.size() % 2 == 1 ? v[v.size() / 2]
                               : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    };
    auto mean_abs = [](const std::vector<double> & v) {
      if (v.empty()) return 0.0;
      double s = 0;
      for (double x : v) s += std::abs(x);
      return s / static_cast<double>(v.size());
    };
    auto max_abs = [](const std::vector<double> & v) {
      double m = 0;
      for (double x : v) m = std::max(m, std::abs(x));
      return m;
    };
    tools::logger()->info(
      "[绿框漂移] n={} | |d| med/mean {:.1f}/{:.1f}px | dx med {:+.1f} dy med {:+.1f} | "
      "droll med {:+.2f}deg mean|.| {:.2f} | dradius med {:+.1f}px mean|.| {:.1f} | "
      "yaw_diff med {:+.2f}deg | R轴端 mean|.| {:.1f}px max {:.1f}",
      stats.green_dr.size(), median_of(stats.green_dr), mean_abs(stats.green_dr),
      median_of(stats.green_dx), median_of(stats.green_dy), median_of(stats.green_droll),
      mean_abs(stats.green_droll), median_of(stats.green_dradius),
      mean_abs(stats.green_dradius), median_of(stats.green_yaw_state),
      mean_abs(stats.green_R_axis_err), max_abs(stats.green_R_axis_err));
    // 第二十二轮: 最差帧定位 —— 按无符号 |d| 排序, 输出 top10 与其分量。
    // 与切板事件日志(帧号)对齐: 若 top 帧都落在切板/断流段附近, 说明是瞬态
    // 而非稳态偏差; 若均匀分布, 说明是持续模型误差, 需要查 R 观测或相位模型。
    {
      std::vector<size_t> idx(stats.green_dr.size());
      for (size_t k = 0; k < idx.size(); k++) idx[k] = k;
      std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return stats.green_dr[a] > stats.green_dr[b]; });
      const size_t n_top = std::min<size_t>(10, idx.size());
      tools::logger()->info(
        "[绿框漂移-top] |d| 最大的 {} 帧 (帧号 t | dx dy | droll dradius | yaw):", n_top);
      for (size_t k = 0; k < n_top; k++) {
        const size_t q = idx[k];
        tools::logger()->info(
          "  帧{} t={:.2f}s |d|={:.1f}px dx={:+.1f} dy={:+.1f} | droll={:+.2f}deg dradius={:+.1f}px | yaw={:+.2f}deg phase={:.1f}deg path={}",
          stats.green_frame[q], stats.green_time[q], stats.green_dr[q],
          stats.green_dx[q], stats.green_dy[q], stats.green_droll[q], stats.green_dradius[q],
          stats.green_yaw_state[q], stats.green_phase[q], stats.green_solve_path[q]);
      }
      // 分位数: 确认误差集中度(p90 vs med)
      std::vector<double> dr_sorted = stats.green_dr;
      std::sort(dr_sorted.begin(), dr_sorted.end());
      const auto q_at = [&](double q) -> double {
        if (dr_sorted.empty()) return 0.0;
        const size_t pos = static_cast<size_t>(q * (dr_sorted.size() - 1));
        return dr_sorted[pos];
      };
      tools::logger()->info(
        "[绿框漂移-分位] p50={:.1f} p75={:.1f} p90={:.1f} p95={:.1f} p99={:.1f} max={:.1f}px",
        q_at(0.50), q_at(0.75), q_at(0.90), q_at(0.95), q_at(0.99), q_at(1.0));
    }
    // 方向判读: dx/dy 中位数给出漂移的主方向; droll vs dradius 判圆周 vs 径向
    tools::logger()->info(
      "[绿框漂移] dx mean|.| {:.1f} max {:.1f} | dy mean|.| {:.1f} max {:.1f} | "
      "droll max {:.2f}deg | dradius max {:.1f}px",
      mean_abs(stats.green_dx), max_abs(stats.green_dx), mean_abs(stats.green_dy),
      max_abs(stats.green_dy), max_abs(stats.green_droll), max_abs(stats.green_dradius));
  }
  // ---- 第八轮: 重复帧检测统计 ----
  // 阈值由预扫描自动标定(第十轮), 此处输出实际判重数与其对照 ——
  // 判重数应与 stsz 小帧数同量级(符.avi ≈168+ 近似重复帧)。
  if (dup_diff_cnt > 0) {
    tools::logger()->info(
      "[重复帧] 总帧 {} | 内容帧 {} | 判重 {} ({:.1f}%) | 差分均值 {:.3f} (阈 {:.2f})",
      stats.frames, content_frame_cnt, dup_frame_skipped,
      100.0 * static_cast<double>(dup_frame_skipped) / static_cast<double>(stats.frames),
      dup_diff_sum / static_cast<double>(dup_diff_cnt), dup_threshold);
  }
  cv::destroyAllWindows();
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);
  auto input_path = cli.get<std::string>(1);
  auto start_index = cli.get<int>("start-index");
  auto end_index = cli.get<int>("end-index");
  auto target_type = cli.get<std::string>("target");
  auto fps_override = cli.get<double>("fps");
  const bool no_stage2 = cli.has("no-stage2");

  // cv::CommandLineParser 只认 `--fps=25`, 不认 `--fps 25`。
  // 写成空格形式时 `--fps` 会被当作布尔标志(值 "true")、`25` 被当作第 3 个位置参数,
  // get<double> 拿到 0 于是静默走"不覆盖"分支 —— 2026-09-09 已经因此白跑一轮
  // (r1/r2 数字逐位相同)。这里让它硬失败, 不要再静默。
  if (!cli.check()) {
    cli.printErrors();
    tools::logger()->error("命名参数必须写成 --名字=值 的形式, 例如 --fps=25 (不能是 --fps 25)");
    return 1;
  }

  if (target_type != "small" && target_type != "big") {
    tools::logger()->error("--target 只能是 small 或 big, 收到: {}", target_type);
    return 1;
  }

  // 开关只接在 SmallTarget::update 上(手上只有小符录像可验证, BigTarget 那一级
  // 没有可对照的数据, 不动它)。所以 --target=big --no-stage2 会是静默空操作 —— 硬失败,
  // 别让人以为关掉了。
  if (no_stage2 && target_type != "small") {
    tools::logger()->error("--no-stage2 目前只对 --target=small 有效");
    return 1;
  }

  if (fps_override < 0 || !std::isfinite(fps_override)) {
    tools::logger()->error("--fps 必须 >= 0 (0 表示不覆盖), 收到: {}", fps_override);
    return 1;
  }

  tools::Plotter plotter;
  tools::Exiter exiter;

  // 允许传 `assets/demo/符.avi`（带后缀）或 `assets/demo/符`（不带, 旧用法）。
  // 先剥掉 .avi 得到公共前缀, 再复用 resolve_runtime_prefix 找到实际位置,
  // 这样从 build/ 目录启动也能定位到 assets/ 下的录像。
  std::string prefix = input_path;
  if (prefix.size() > 4 && prefix.compare(prefix.size() - 4, 4, ".avi") == 0)
    prefix.erase(prefix.size() - 4);

  auto video_path = tools::resolve_runtime_path_string(fmt::format("{}.avi", prefix));
  auto text_path = tools::resolve_runtime_path_string(fmt::format("{}.txt", prefix));

  cv::VideoCapture video(video_path);
  if (!video.isOpened()) {
    tools::logger()->error("无法打开录像: {}", video_path);
    return 1;
  }

  // 四元数日志可选。没有就退化为单位四元数(云台系 == 世界系)。
  std::ifstream text(text_path);
  const bool has_quaternion = text.is_open();
  // 没有四元数日志时用帧率推算时间戳。EKF 的 dt 依赖它, 取不到就退回 30 fps。
  double fps = video.get(cv::CAP_PROP_FPS);
  if (!(fps > 1.0) || !std::isfinite(fps)) {
    fps = 30.0;
  }

  // --fps 覆盖。用途见诊断文档 17.9 候选 1：步骤 17 量出 符.avi 的 roll 表观转速
  // 是 120 deg/s (2.413 deg/帧), 正好是小符假设值 SMALL_W=60 的 2.011 倍。
  // 容器头自洽地声明 50 fps, 所以「录像真的转 2 倍」与「真实采集间隔是 40 ms」
  // 从文件里分不开 —— 但两者都预测: 把时间基准减半后故障 3 应完全消失。
  // 这是个可否证的预测: 若撞门次数没归零, 说明 spd 上还有第二个机理。
  // 只影响测试推算时间戳, 生产代码用相机时间戳, 不受此参数影响。
  if (fps_override > 0) {
    fps = fps_override;
  }


  // --no-stage2: 只为把两个候选机理分开而存在的诊断开关, 不是修法。
  //
  // 关掉第二级更新后, 转轴 (R_yaw/R_pitch/R_dis) 只由第一级的 R 标直接观测决定,
  // 而 R 标观测的实测精度是 3.0 px。所以判据很干脆:
  //   ekf_r_err 掉到 ~3 px  → 那 117 px 是阶段 2 每帧推出来的, 责任在 pitch=0 的结构性误差
  //   ekf_r_err 仍是 ~117   → 阶段 2 无关, 误差是 init 那几帧留下、被零 Q_ 锁死的
  // 注意这一路 yaw(x[4]) 会失去观测、停在 init 值, 绿色四边形的朝向因此不可信 ——
  // 但用户判据只看轴线端点(image_points[6]), 它只依赖 x[0]/x[2]/x[3], 不受影响。
  if (no_stage2) {
  }

  if (target_type == "small")
    run<auto_buff::SmallTarget>(
      video, video_path, text, has_quaternion, fps, config_path, start_index, end_index, plotter,
      exiter, no_stage2, false);
  else
    run<auto_buff::BigTarget>(
      video, video_path, text, has_quaternion, fps, config_path, start_index, end_index, plotter,
      exiter, no_stage2, true);
  if (text.is_open()) text.close();
  return 0;
}

# Buff 能量机关模块 — 结工汇总文档

最后更新：2026-09-15（小符+大符收工；大符已接入 auto_buff_debug_mpc 主链路）。本文是 `buff_model/` 唯一主文档，
覆盖全模块核心代码设计与迭代要点，供后续优化快速上手。调试过程细节已精简，
只保留影响当前行为的定案结论。

**代码位置**：`sp_vision25-debug/tasks/auto_buff/`（生产代码）
+ `tests/auto_buff_test.cpp`（离线验证）+ `src/`（正式入口）。
Windows 侧改代码，Linux 侧编译验证（`cmake --build build -j`）。

**绝对铁律**：`buff_target.cpp` 前 474 行 = SmallTarget（小符），逐字零改动；
`detect()` / `get_onecandidatebox()` 小符路径不碰。大符一切改动走旁路。

---

## 1. 数据流总览

```
图像 → YOLO11_BUFF 推理解码(5关键点+3分类+R标)
  → Buff_Detector 选叶
      小符: detect() 单候选(class0最高分)
      大符: detect_big() 多候选 + 身份锁定状态机
  → PowerRune(观测) → Solver::solve() PnP位姿 + R标锚定 + 几何相位
  → Target EKF
      小符: SmallTarget 7态(恒速模型)
      大符: BigTarget 11态(正弦转速模型, b自由状态)
  → Aimer::aim() 前瞻predict + 两次弹道迭代 → 云台指令
```

分层可视化约定（白/绿/蓝）：白线=纯PnP位姿（几何层），绿框=EKF当前状态重投影
（滤波层），蓝框=前瞻预测重投影（预测层）。三层叠加时"哪层偏就是哪层的问题"。

---

## 2. 检测层（yolo11_buff.{hpp,cpp}）

### 模型契约
- IR：`Rune-v8n-fp16-20260624-D14367-B16`（OpenVINO，CPU 实测可用）
- 输入 1×3×480×640（H=480 W=640，letterbox 4:3 无 padding）
- 输出 18×6300：ch0-2 三分类分数，ch3-17 为 5 关键点×(x,y,conf)
- 关键点顺序 top/left/**R**/right/bottom，索引 2 是 R 标（转轴中心）
- 类别：0 未激活 / 1 小符已激活 / 2 大符已激活
- 关键点重排 `kpt={raw0,raw1,raw4,raw3}` 对应 OBJECT_POINTS 前四点，数值不改
- 启动自检：`output shape=[1,18,6300]`，形状不对直接异常（防新旧模型混用）

### 两个候选入口
- `get_onecandidatebox()`：小符。只在 class0 里取最高分，无 class0 返回空
  （不退回最高分——那等于瞄准已激活叶）。class0 过滤消除帧间换叶。
- `get_multicandidateboxes()`：大符。全候选上抛 + **环带门**过滤臂灯杂点：
  候选中心到 R 距离 / 四角点基线 ∈ [1.62, 3.61]（装甲环带 1.97~3.26 ± 0.35），
  基线 <5px 直接拒。**锁定邻域救援**：detect_big 在 decode 前通报上一帧锁中心，
  被环带门拒但中心落在锁中心 40px 内的候选放行（防误杀真叶断流）。
- `DecodeStats`：候选全景 + 锁定状态机事件 + `mark_chosen()/set_lock_event()`
  供上层回填真实选中项（取证必须量"实际选中的叶"，不是 objects[0]）。

---

## 3. 选叶与身份锁定（buff_detector.{hpp,cpp}，大符 detect_big）

设计原则：**跟踪转子的连续相位，不跟踪"某片叶"**——叶身份是离散事件，转子相位
才是连续可滤波量。

锁定状态机（八态事件：NONE/KEEP/MISS1/RELEASE/GHOST_HIT/GHOST_MISS/LOWER/COLDSTART）：
1. **已锁定 → 近邻续锁**：候选中距上帧锁中心最近者；门 80px（同叶单帧位移
   ~7px @97fps，相邻叶心间距远大于此）
2. **连续 ≥2 帧找不到才释放**（单帧掉线只 miss 不切，宁空过不喂错叶）
3. **释放后幽灵锁**：记 ghost_center + TTL 8 帧（~83ms@97fps），重捕获先在
   幽灵位 60px 内找回原叶（防"光线抖两帧→永久切板"）；TTL 尽走偏下规则
4. **冷启动重捕获 → 偏下规则**：相对融合 R 标 dy 大者（低头平稳、云台路径短），
   高度差 <15px 平局带内按置信度
5. `phase_hint_rad` 参数保留但已不参与选叶（早期相位槽位法因 a/ω/φ 未收敛期
   相位漂移形成自激循环而废弃）

R 标多候选中位数融合：所有候选的 R 关键点取中位数作为唯一 R 观测。

---

## 4. 解算层（buff_solver.{hpp,cpp}）

### 4.1 PnP + IPPE 二义性
- 7 点 OBJECT_POINTS 全共面（x=0）：4 角点(254mm基线) + 叶心700mm + 中间220mm + R标原点
- `solvePnPGeneric(IPPE)` 取全部解，按"与上一帧姿态最近"挑（Rodrigues 模长），
  成对变号 151→0。共面二义性只能靠时序挑解，加共面点消不掉。
- R 标过几何门后作第 5 点进 PnP + `solvePnPRefineLM` 精化；不过门退 4 点
  （`last_solve_used_r()` 可查路径）。R 进 PnP 治的是 254mm 基线病态性。

### 4.2 R 标几何门（纯 2D，不依赖位姿）
- `t_along = R沿上下延长线位置/基线` ∈ [1.2, 3.2]
- `d_perp = 横向偏移/基线` < 0.40；基线 <5px 拒收

### 4.3 R 像素硬锚定（转轴三层锚定的第二层）
PnP 后、输出 EKF 前：`undistortPoints(R像素) → r_hat 射线`，深度取
`t_buff2camera · r_hat`。方位 2 自由度由 R 像素直接给定（σ≈0.001rad），
深度由 PnP 沿射线分量给定。与 R 进 PnP 同一道几何门。

### 4.4 几何相位 blade_phase（绕开欧拉分解，根治 IPPE 泄漏）
```
col0 = R_buff2world 第0列(转轴/法向, 只含yaw)
col2 = R_buff2world 第2列(叶心方向, 只含roll)
ê   = 世界竖直(0,0,1)投影到旋转平面
phase = atan2((col0×ê)·col2, ê·col2)
```
与欧拉 roll 数学等价但无奇异性、IPPE 翻转时 yaw 项被消去不污染相位。
退化（相机正对转轴 e_hat.norm()<1e-9）退回欧拉 roll 兜底。
**EKF roll 观测与转速仪表全部以 blade_phase 为源。**

### 4.5 reproject_buff（绿/蓝框重投影）
R_len/C_len 人工微调在此生效：径向点绕转轴旋 θ=C_len/(0.7·R_len) 后 z 乘 R_len。
原点（R 标）不受影响，轴端永远钉死 R 十字。Aimer 的瞄准点与这里的重投影共用
同一套参数（从 aimer.R_len()/C_len() 取），绿框/蓝框/实际打弹三者联动。

---

## 5. 滤波层（buff_target.{hpp,cpp}）

### 5.1 基类 Target
- `point_buff2world()`：Rz(yaw)·Rx(roll)，**pitch 恒 0**（已知模型假设）
- `Voter`：转向投票（小符在用；大符因回绕错票+符号过零翻转已不用，见 §5.3）

### 5.2 SmallTarget（前 474 行，铁律不动）
7 态：[R_yaw, v_R_yaw, R_pitch, R_dis, yaw, roll, spd]，恒速模型。
- R 状态硬锁定：update 尾直接回拷观测、v_R_yaw=0（转轴是世界系定点）
- spd 过程噪声 0.01、发散门 |spd|>4 或 <π/36、差分初值（最近 8 条 roll 历史）
- 切板防污染：叶位跳变对齐后切断 roll-spd 互协方差、P[5][5] 放宽
- 小符离线定案结果：轴端 0px、spd ~61°/s(官方 10RPM)、绿蓝夹角 ~26°
  全部有物理来源

### 5.3 BigTarget（大符，line 474 起）
**11 态**：[R_yaw, v_R_yaw, R_pitch, R_dis, yaw, roll, spd, a, ω, φ, **b**]
转速模型：`spd = a·sin(ωt+φ) + b`，b=x[10] 自由状态（b 状态化，2026-09-15）。
- 自制匀速慢转机关：a→0、b→真实转速，预测蓝框与绿框保持稳定小偏移
- 官方转速数据：自然收敛到 b≈2.090−a，与旧模型数学等价
- **无任何"慢速/快速"模式开关**——预测逻辑本身通用，这是设计约束不是缺陷
- ω 物理边界 [0.3, 3.0] rad/s（通用，非官方包络）；a 发散门 [0, 1.6]
- b 不回中点：会话内平均转速是缓变量，参数重置只重摆动项(a/ω/φ)

**闭式预测**（不靠数值积分）：
```
spd_at(x,t) = x[7]·sin(x[8]·t+x[9]) + x[10]
roll_at(x,t0,t) = x[5] + sgn·[a/ω·(cos(ωt0+φ)−cos(ωt+φ)) + b·(t−t0)]
```
`roll_diff_jacobian()` 为解析雅可比。`phase_t0_` 记状态有效时刻（aim 前瞻
predict 会把状态推到未来，update 必须以它为积分起点，不能复用 lasttime_）。

**update() 流程**（顺序敏感）：
1. 槽位对齐：|phase−roll|>15° 时在 ±5 槽位(72°)找最近对齐；对齐后
   P[5][5]=10、切断与 a/ω/φ 互协方差、**置 track_r1_tight_=40**（方案C：
   对齐后 40 帧 R1 相位噪声 0.1→0.02 加速咬回）
2. 跳变率检测：1s 窗口跳变>30% → reset_sine_params()（只重置 spd/a/ω/φ
   子块，保留 R 系与转子相位——会话边界转子连续）
3. **转向锁存**（第二十三轮）：相邻观测相位差 limit_rad 投票（回绕安全、
   不掺 EKF 残差），|差|>36° 切板帧弃票，净票 ±20 永久锁定 dir_sign_。
   锁定后 roll_at/雅可比的 sgn 读锁存值，不再读 Voter（Voter 回绕错票+
   符号过零翻转是"蓝框在绿框前后跳"的根因）
4. dt_meas 预测（**第二十五轮根治点**）：`Q = Zero(10,10) 会让 P 几何塌缩
   → K→0 → yaw 冻结**（绿框右下漂根因）。已补按 dt 比例过程噪声：
   Q(4,4)=0.09·dt, Q(5,5)=0.09, Q(6,6)=0.05, Q(7,7)=0.002, Q(8,8)=0.0005,
   Q(9,9)=0.05。**教训：测试链路 aim 作用于副本 target_copy，真身只走
   这条 Q 路径；任何 predict 路径的 Q 都不能为 0，否则状态冻死**
5. 第一级观测 z1（5 维）：[R_yaw, R_pitch, R_dis, blade_phase, **PnP yaw**]
   —— yaw(x[4]) 直接受观测（第二十一轮加，治 yaw 无直接观测漂移）
6. 第二级叶心观测（h_jacobian + point_buff2world）
7. ω 硬 clamp [1.884, 2.000]（规则包络，防 predict 积分放大）
8. **R 锚定 EWMA**：锚定计数<5 硬锁定快速建立；之后观测先过离群门
   （10°/10°/0.5m）拒收坏帧，通过者 α=0.1 EWMA 融入，锚定值写回 x[0..3]
9. RansacSineFitter 只读仪表（拟合结果不回流 predict，防双模型拉扯）
10. track_r1_tight_ 递减（在 R1 用过之后，保证恰好 40 帧）

**keep-alive**（第十八轮）：无观测帧 lost_cn_≤5 且空窗<0.3s 时纯 predict
推进、unsolvable_ 保持 false（绿框不闪）；超窗才 unsolve。整体重初始化门
lost_cn_>6（注：第十六轮曾改 16 适配 97fps，后随第十七轮回退 ghostwait
一并回退为 6；97fps 录像下幽灵窗最坏 10 帧会触发整体重初始化——若
高帧率场景再现蓝框乱闪，此处是第一嫌疑）。
time_point 时间戳链（now_tp_→last_update_tp_→lasttime_tp_）与 double 型
lasttime_ 分离（wall-clock 空窗判定与 EKF 时间基不同源）。

### 5.4 已知模型局限
- `point_buff2world` pitch=0 硬编码：机关若俯仰倾斜有结构误差
- 自制机关适配已由 **b 状态化**根治（不再需要硬编码官方包络或 yaml 参数）
- 方案 C（track_r1_tight_）实测对绿框漂移贡献有限（根因是 Q=0），保留无害

---

## 6. 瞄准层（buff_aimer.{hpp,cpp}）

- `aim(target, timestamp, bullet_speed, to_now)`：
  lead = (检测延迟 + predict_time) × buff_predict_lead_scale，**弹道飞行时间
  是物理量不缩放**。get_send_angle 两次弹道迭代（Trajectory 类）。
- `mpc_aim()`：`buff_mpc_feedforward=false` 规避已知前馈缺陷
  （yaw_vel 符号反 / pitch_acc 公式错 / 开火无对准检查——三处既有缺陷
  未修，另立任务）
- 换叶判定：瞄准角跳 >5° 抑制开火（mistake_count_>3 强制 control）
- R_len/C_len 从 Aimer 读取（与重投影联动），支持 runtime_params 热更新
- 大符开火策略（低速窗口等）**未实施**——当前仍是小符式开火节奏

---

## 7. 离线验证链路（tests/auto_buff_test.cpp）

```bash
cd sp_vision25-debug && cmake --build build -j
./build/auto_buff_test configs/standard3.yaml assets/demo/符 --target=big
# 小符: 同命令去掉 --target=big，必须带 --fps=25（容器头错标50，真实25）
```

- 录像旁同名 .txt = 四元数日志（相机姿态），无则单位四元数；日志行数 < 录像
  帧数时提前终止回放（防假时间戳污染 EKF）
- **判重**：预扫描独立句柄自动标定阈值（双峰谷底法）；重复帧 t 钉在上一
  内容帧（dt=0）；健康录像（近零帧<10%）自动禁用
- 可视化：白线 PnP / 绿框 EKF / 蓝框预测 + 转向锁存日志 + 切板事件日志
  （10 帧环形缓冲，事件触发打印上下文）
- **绿框漂移仪表**（第二十~二十五轮，判定修复效果的核心工具）：
  green_blade(EKF 700mm 点) vs pnp_blade(PnP 700mm 点) 分解为
  droll/dradius/dx/dy/yaw_diff/R_axis_err + top10 帧（含 PnP 解算路径
  path 列）+ 分位数（p50~max）。判读：droll大→相位；dradius大→转轴/yaw；
  R_axis_err大→R锚定坏了
- aim 作用于 **target_copy 副本**——真身 EKF 不吃 aim 的 predict，
  验证滤波可观性时要记得这条链路差异

## 8. 正式入口与生产链路

- `src/auto_buff_debug_mpc.cpp`：单人调试入口（mpc_aim）。**已接入大符全链路**
  （`--target=big`，默认 small 行为零改动）：detect_big 多候选+身份锁定 →
  BigTarget 11 态 EKF → mpc_aim 前瞻，phase_hint 传上一帧 EKF 转子相位。
  小符/大符各持独立 target 对象，经 `Target*` 基类指针统一调度；副本化
  （small_copy/big_copy）与测试链路同构，aim 的 predict 只推副本。
- `src/mt_standard.cpp`：多线程标准入口。大符模式 **line 115 仍调 detect()**
  （未接 detect_big）——身份锁定/幽灵锁/环带门目前只在测试链路与
  auto_buff_debug_mpc 生效，见 §11 遗留事项 1
- `buff_record`（tests/buff_record.cpp）：MJPG 录制工具 + 四元数同步日志，
  帧间间隔自检。用法：`./build/buff_record configs/standard3.yaml
  --output=assets/demo/名字 --seconds=60`

---

## 9. 关键 yaml（configs/standard3.yaml，buff 相关）

| 键 | 当前值 | 说明 |
|---|---|---|
| model | assets/Rune-v8n-fp16-...-B16.xml | 六个 yaml 全部同指 |
| buff_confidence_threshold | 0.5 | 三分类分数阈 |
| buff_keypoint_threshold | 0.8 | 单关键点阈 |
| buff_min_valid_keypoints | 3 | 达标关键点下限 |
| buff_center_dist_threshold | 100 | centerNMS px |
| buff_predict_lead_scale | 0 | 提前量缩放（0=无提前，调试值）|
| buff_mpc_feedforward | false | 前馈已知缺陷规避 |
| R_len | 0.98 | 绿框径向缩放（轴端不动）|
| C_len | 0.0 | 绿框圆周弧长偏移 m |

R_len/C_len/R_gimbal2imubody/camera_matrix 等支持 runtime_params 热更新
（tools/runtime_params.cpp 注册表，web 调参）。

---

## 10. 迭代史要点（只列影响现状的定案）

**小符（08-31~09-12，已定案）**
1. 模型从 yolo11_buff_int8 换 Rune-v8n（拿 R 标、多候选）→ 检测层全重写
2. R 转轴三层锚定（PnP 第5点 + 像素硬锚 + EKF 状态回拷）：轴端 134px→0
3. class0 过滤消帧间换叶；IPPE 时序挑解消成对变号
4. 时间基定案：测试必须 --fps=25；spd 通道修复；真雷（Q_ 越界写等）
5. R_len/C_len 绿框微调参数（与 yaw/pitch_offset 平移补偿正交）

**大符（09-13~09-15，本轮收工）**
1. big 旁路链路：detect_big + BigTarget EKF（时为 10 态，第 8 条升 11 态）+ 测试入口
2. 身份锁定状态机取代相位槽位法（自激循环根因）
3. 几何相位 blade_phase 根治欧拉分解泄漏（raw_roll 隔帧 -300°/s 假观测）
4. 录像质量链：判重预扫描 + buff_record 工具 + 日志同步耗尽即停
5. 幽灵锁防"光线抖两帧永久切板"；lost_cn_ 门 6→16 适配 97fps
6. 识别稳定性四件套：R 锚定 EWMA+拒收 / keep-alive 防绿框闪 /
   环带门救援 / 切板取证日志重做
7. 绿框漂移五连修：yaw 直接观测(治中位偏置) → 最差帧仪表(定位瞬态) →
   转向锁存(治蓝框跳变) → 方案C(无效但保留) → **Q=0 根治(yaw 冻结根因)**
8. **b 状态化（自制机关适配定案）**：spd 常数项 b 提升为自由状态 x[10]，
   10 态→11 态。自制匀速慢转 a→0/b=真实转速，官方数据 b→2.090−a，
   不引入任何模式开关；ω 边界改通用物理范围 [0.3,3.0]
9. **auto_buff_debug_mpc 接入大符**：--target=big 走 detect_big+BigTarget
   全链路，小符默认路径逐位不变
10. 最终状态：用户确认绿框贴合真实叶片、匀速慢转蓝框稳定领先，任务收工

**作废判据（别再用）**
- PnP 重投影 RMS<3px（恰定问题残差恒≈0，无鉴别力）
- buff_R_dis 跨轮比较（fp16 非确定性+深度病态，漂 6%）
- 加共面点消 IPPE 二义性（数学上不可能）
- 「spd 稳定偏向=滤波缺陷」（真因曾是中国时间基标量错）

---

## 11. 遗留事项（不阻塞，接手时优先看）

1. **mt_standard.cpp 生产链路大符仍未接 detect_big**（L115 仍调 detect()）——
   大符全链路现已在测试链路 + auto_buff_debug_mpc 生效，多线程标准入口是
   最后一个未接入点（改法参照 auto_buff_debug_mpc.cpp 现成模式）
2. Aimer 前馈三缺陷（yaw_vel 符号 / pitch_acc 公式 / 开火无对准检查）
3. 大符开火策略未做（低速窗口 |spd|<阈值、参数收敛门）
4. `point_buff2world` pitch=0 假设（机关俯仰倾斜时有结构误差）
5. ~~自制机关转速模型参数硬编码官方值~~（已由 b 状态化根治，见 §5.3；
   备份 `buff_target.cpp.bak-20260915-bfree` 记录改造前状态）
6. QUICK FORENSIC LOG（buff_target.cpp update 内每帧打 phase/roll/d）
   量大，修复已验证，可删或降级 debug
7. 旧录像 符.avi/符2.avi 是坏的（30% skip 帧填充），用 buff_record 重录的
   MJPG（符/）才是有效验证素材

---

## 12. 方法论（压成五条）

1. **先仪表后修法**：只读仪表定位到状态分量再动手；中位数好≠修好，
   要看 top/分位数暴露长尾
2. **测试链路≠生产链路**：副本化(aim→target_copy)会切断某些生产路径
   （Q>0 的 predict），排查滤波问题先列出全部调用路径
3. **改前备份，改后三重复查**：SmallTarget 逐字 diff / 括号配对 /
   改动点后尾部位移对齐。PowerShell 批量正则改 C++ 高危（本项目两次误伤）
4. **几何直觉不算证据**：因果假设要设计成可否证的并真的去否证
5. **先怀疑输入再怀疑滤波器**：估计量稳定偏向某方向时先量真值
   （时间基/录像质量都在这一类）

# sit — 坐姿检测助手

基于人体关键点几何角度的实时坐姿检测服务，后端为纯 Golang 实现。

```
┌─────────────────────────┐   WebSocket（33 个关键点）    ┌──────────────────────┐
│  浏览器                  │ ───────────────────────────▶ │  Go 后端              │
│  MediaPipe Pose          │                              │  posture 判定引擎      │
│  Landmarker（本地推理）    │ ◀─────────────────────────── │  角度计算/去抖/久坐计时 │
│  摄像头画面 + 骨架叠加     │      判定结果 + 角度指标       │  配置 API              │
└─────────────────────────┘                              └──────────────────────┘
```

视频画面只在本机浏览器中分析，服务器只接收 33 个归一化骨架坐标。

## 检测能力

角度判定基于**个人标定基线**：首次使用时保持标准坐姿标定一次（页面按钮，采集 3 秒），
之后各项角度指标都以“相对标定坐姿的偏差”参与阈值比较，摄像头摆放角度带来的
固有偏差会被自动抵消。基线持久化在 `~/.sit-calibration.json`（`-calibration` 可改路径），
服务重启无需重新标定；页面按钮可随时重新标定。

**未标定的视角不参与角度判定**（状态返回 `needs_calibration`），久坐计时照常工作。

### 单 / 双摄像头

- **单摄像头**（默认）：所有判定共用一路画面，前倾类指标依赖单目深度估计，建议摄像头放在斜前方约 45°。
- **双摄像头**：检测到两个摄像头时自动进入双视角模式——**正视摄像头**负责头部侧倾（冠状面），
  **侧视摄像头**负责头部前倾与躯干前倾/后仰（矢状面，纯 2D 平面角，无深度噪声，精度最高）。
  两个视角各自标定、各自去抖，久坐计时按“任一视角看到人”全局合并（交错帧不重复计时）。

| 问题 | 判定视角 | 判定依据（相对标定坐姿的偏差） | 默认阈值 |
|---|---|---|---|
| 头部前倾 `forward_head` | 侧视/单目 | 肩中点→耳中点连线与垂直轴的矢状面夹角偏差 | 轻度 ≥ 20°，重度 ≥ 35° |
| 身体前倾 `torso_lean_forward` | 侧视/单目 | 髋中点→肩中点连线与垂直轴的矢状面夹角偏差 | 轻度 ≥ 20°，重度 ≥ 35° |
| 身体后仰 `torso_lean_back` | 侧视/单目 | 同上（反方向） | 轻度 ≥ 20°，重度 ≥ 35° |
| 头部侧倾 `head_tilt_left/right` | 正视/单目 | 双耳连线与水平线的夹角偏差 | 轻度 ≥ 12°，重度 ≥ 25° |
| 久坐 `long_sitting` | 全局 | 连续在座时长（离座暂停，超过 2 分钟重新计时） | 45 分钟 |

标定帧取各指标在采集窗口内的**中位数**，个别抖动帧不会污染基线。
角度判定带去抖状态机：连续 8 帧超标才告警、连续 15 帧恢复才解除，避免抖动误报。
髋部不在画面时自动跳过躯干类判定（只监控上半身也可用）。

### 语音提示

状态卡右侧的「语音提示」开关（默认关闭，状态保存在浏览器本地）打开后，坐姿出现
问题时用浏览器内置语音（Web Speech API，中文）播报提醒：

- 问题出现时播报一次，多个问题合并成一句话
- 轻度恶化到重度时再播报一次，重度问题会带上“明显”字样
- 持续不良期间每 2 分钟重复提醒；恢复后重置，再次出现会重新播报
- 纯前端实现，无音频文件依赖；语音文案与问题列表共用同一份映射表

## 运行

```bash
go run .
# 打开 http://localhost:8080/ ，允许摄像头授权
# 首次使用：摆好标准坐姿 → 点「开始检测」→ 点「标定正确坐姿」（保持 3 秒）
```

浏览器需要能访问两个 CDN：`@mediapipe/tasks-vision`（JS/WASM 运行时）与
Google Storage（姿态模型 `pose_landmarker_lite.task`，约 5.5 MB，首次加载后由浏览器缓存）。

标定后摄像头角度不再敏感：可以正对摄像头使用；只要标定时的机位与日常使用一致即可。

## HTTP / WebSocket API

| 接口 | 说明 |
|---|---|
| `GET /` | 内置检测页面 |
| `GET /api/config` | 读取当前阈值配置 |
| `PUT /api/config` | 整体替换阈值配置（校验失败返回 400，拒绝未知字段） |
| `POST /api/calibrate` | 标定一个视角：`{"view":"front","poses":[{"landmarks":[...],"timestamp_ms":...}]}`，view 取 `front`/`side`/留空（单目 default）；有效帧不足 8 帧返回 400 |
| `GET /api/calibration` | 各视角标定状态 `{"views":{"front":{"calibrated":true,"baseline":{...}}}}` |
| `DELETE /api/calibration?view=front` | 清除指定视角（不带参数清除全部）的基线（内存 + 持久化文件） |
| `GET /ws/pose` | 坐姿流：客户端发 `{"view":"side","landmarks":[{"x","y","z","visibility"}×33],"timestamp_ms":...}`，服务端回全局快照 `{"status","view","metrics","issues","seated_seconds","views"}` |

坐姿流帧的 `view` 决定该帧参与哪些判定：`front` 只判头部侧倾，`side` 只判前倾类，
留空/`default` 为单目模式全量判定。响应中的 `issues` 是所有视角活跃问题的并集
（每个问题带 `view` 标签），`views` 是各视角的在座/标定/状态摘要，`metrics`
为该帧所属视角的指标（不归属该视角的指标置 0）。

`status` 取值：`good` / `warning` / `bad` / `no_person` / `needs_calibration`（在座视角未标定）。
`issues[].code` 为稳定枚举（上表第一列），客户端自行维护文案映射；
后端不返回展示文本，未知 code 由前端兜底显示。

配置示例：

```bash
curl -X PUT localhost:8080/api/config -H 'Content-Type: application/json' -d @- <<'EOF'
{
  "min_visibility": 0.5,
  "neck_forward_mild_deg": 20, "neck_forward_severe_deg": 35,
  "torso_forward_mild_deg": 20, "torso_back_mild_deg": 20, "torso_severe_deg": 35,
  "head_tilt_mild_deg": 12, "head_tilt_severe_deg": 25,
  "long_sitting_minutes": 45,
  "hold_frames": 8, "clear_frames": 15,
  "absence_reset_seconds": 120
}
EOF
```

## 代码结构

```
posture/    判定引擎（纯 Go，无外部依赖）
  pose.go          33 关键点数据结构与可见性判定
  geometry.go      角度计算（矢状面/冠状面投影）
  calibration.go   标定基线（中位数）计算
  config.go        阈值配置与校验
  engine.go        基线偏差判定 + 去抖状态机 + 久坐计时
server/      HTTP 层
  server.go        路由、配置 API
  calibration.go   标定 API + 基线持久化
  ws.go            WebSocket 流 + 内嵌页面资源
  web/             内置前端（原生 JS + MediaPipe）
main.go      入口
```

设计为单人监测场景：全局共享一个判定引擎，去抖与久坐状态即“当前这个人”的状态。

## 验证

```bash
go build ./...
go test ./...
go vet ./...
```

## 方案来源与参考

检测方法沿袭开源坐姿检测项目的通用做法（姿态估计关键点 + 几何角度阈值 + 去抖）：

- Google [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)（Apache-2.0）：33 个 3D 关键点，浏览器端 WASM 推理
- [pose-nudge](https://github.com/DDULDDUCK/pose-nudge)（190★）：摄像头检测头部前倾并提醒
- [Sitting-Posture-Recognition](https://github.com/nvinayvarma189/Sitting-Posture-Recognition)（173★）：OpenPose 关键点坐姿分类
- [sitting-posture-detection-yolov5](https://github.com/itakurah/sitting-posture-detection-yolov5)（95★，IEEE BioRob 2024，DOI 10.1109/BioRob60516.2024.10719953）：YOLOv5 侧面坐姿检测
- [upright](https://github.com/LinklyAI/upright)（32★）：纯浏览器实现，检测驼背/前倾/歪头/耸肩/久坐

Go 生态没有成熟的姿态估计库，因此关键点提取放在浏览器（MediaPipe 官方 WASM），
后端专注判定逻辑。若后续需要接入 IP 摄像头做纯后端推理，可保持
`posture.Engine` 不变，新增一个产出 `posture.Pose` 的推理源
（如 [yalue/onnxruntime_go](https://github.com/yalue/onnxruntime_go) + YOLOv8n-pose ONNX）。

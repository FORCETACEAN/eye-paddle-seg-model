# EyePaddleSeg — 遮眼板实例分割模型

单类"遮眼板"（eye occlusion paddle）实例分割模型，用于驾驶员视力检测设备中
判断被测眼是否被遮眼板正确遮盖。基于 Ultralytics **YOLO26n-seg**（AGPL-3.0）
在 192×192 眼部 ROI 图上训练，端到端导出 ONNX（内置 NMS，推理侧无需后处理代码）。

> **License: AGPL-3.0**（见 [LICENSE](LICENSE)）。本模型使用 Ultralytics YOLO26
> 训练，按 Ultralytics 官方口径，训练产物属 AGPL-3.0 衍生物。
> 如需闭源商用授权，请联系 Ultralytics：https://www.ultralytics.com/license

## 模型文件

| 文件 | 说明 |
|---|---|
| `eye_paddle_seg.onnx` | 约 10.5 MB，opset 12，端到端导出（nms=True） |
| `train_formal.py` | 训练脚本（Ultralytics API 调用） |
| `dataset.yaml` | 数据集配置模板（**数据集本身为专有数据，不在本仓库分发**） |

## 推理接口

**输入** `images`：`[1, 3, 192, 192]` float32，RGB 顺序，数值 0~1，NCHW。
内容为按固定规则从 1080p 原图裁剪的 192×192 单眼 ROI（裁剪依赖 MediaPipe
FaceLandmarker 关键点，face_landmarker.task 为 Apache-2.0，不在本仓库范围）。

**输出**（2 个张量，坐标均在 192 ROI 像素域）：

| 张量 | shape | 含义 |
|---|---|---|
| `output0` | `[1, 300, 38]` | 300 行候选，每行 `[x1, y1, x2, y2, conf, cls, 32 个 mask 系数]`；低分补位行全 0，端到端已内置 NMS，直接取 conf 最高行 |
| `output1` | `[1, 32, 48, 48]` | mask 原型（proto），原始值需过 sigmoid |

**mask 解码**：

```python
import cv2
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("eye_paddle_seg.onnx",
                            providers=["CPUExecutionProvider"])
out = sess.run(None, {"images": img[None]})   # img: (3,192,192) float32 RGB 0~1
dets, proto = out[0][0], out[1][0]            # (300,38) / (32,48,48)

i = int(np.argmax(dets[:, 4]))                # 取置信度最高的一行
conf, box, coef = dets[i, 4], dets[i, :4], dets[i, 6:38]
m = 1 / (1 + np.exp(-(proto.reshape(32, -1).T @ coef))).reshape(48, 48)
mask = cv2.resize(m, (192, 192))              # 连续值 0~1
mask_bin = mask >= 0.5                        # True = 模型认为该像素属于板面
```

> 注意：自行用 Ultralytics 重导出时必须带 `nms=True`（`yolo export ... nms=True`）。
> 默认导出（nms=False）为标准双张量 `[1,37,756]` 格式，与上述解码不兼容。

## 训练

- 预训练权重：`yolo26n-seg.pt`（COCO，迁移学习）
- 数据：192×192 眼部 ROI，单类 `eye_paddle`，多边形标注沿板面真实弧形边缘
- 训练参数见 [train_formal.py](train_formal.py)：imgsz=192、batch=32、epochs=200、
  patience=50、seed=42、Mosaic=1.0（close_mosaic=30）、HSV/几何增强
- 本仓库发布的权重：best epoch=160，验证集（133 张 ROI）P=0.9986 / R=1.0、
  Mask mAP50=0.995 / mAP50-95=0.9733

## 重新分发

依 AGPL-3.0，你可以自由使用、修改、再分发本仓库内容，但必须向你的接收者
提供同等许可与完整对应源码（含修改）。本仓库的训练数据集不在 AGPL 义务范围内，
不随再分发提供。

# -*- coding: utf-8 -*-
"""EyePaddleSeg 正式训练脚本（单类遮眼板实例分割，YOLO26n-seg）。

迁移学习：从官方预训练权重 yolo26n-seg.pt（COCO）起步，禁止从零训练。
训练数据为 192×192 眼部 ROI 图（YOLO-seg 格式，单类 eye_paddle），
数据集本身不在本仓库中，路径通过 dataset.yaml 指定。

Windows 注意：spawn 启动的 dataloader worker 会重新执行本文件
（此时 __name__ != "__main__"），重型 import 必须放在保护内，
避免每个 worker 进程重复加载 torch/CUDA DLL。
本机另有一处环境限制：页面文件不足时任何 dataloader worker 都会
触发 WinError 1455，因此 workers=0（同步加载），不影响最终模型。
"""
if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO("yolo26n-seg.pt")
    model.train(
        data="dataset.yaml",
        imgsz=192,          # 与部署侧 ROI 输出尺寸严格一致
        batch=32,
        epochs=200,
        patience=50,
        lr0=0.005,          # 小数据防前期震荡（optimizer=auto 下实际由 AdamW 调度覆盖）
        degrees=10,         # 几何增强：现场姿态变化
        translate=0.1,
        scale=0.3,
        hsv_h=0.015,        # 光照增强：v（亮度）抖动最重要
        hsv_s=0.6,
        hsv_v=0.5,
        fliplr=0.5,         # 单类不区分左右 → 翻转合法
        flipud=0,           # 人脸上下不对称，禁垂直翻转
        mosaic=1.0,
        close_mosaic=30,    # 最后 30 epoch 关 Mosaic，分布回归真实
        seed=42,
        device=0,
        workers=0,
        project="runs/eye_paddle",
        name="v1",
    )

import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile


def export_to_mobile():
    print("1. 正在加载预训练模型...")
    # MobileNetV2 是非常经典的端侧轻量级模型。
    # pretrained=True 代表我们不仅加载了网络结构（骨架），还下载了已经在 ImageNet 数据集上训练好的权重（灵魂）。
    model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # 【Infra 必考点 1：模型状态切换】
    # 将模型设置为评估模式（推理模式）。
    # 这一步极其重要！它会关闭 Dropout 层（防止推理时随机丢弃神经元），
    # 并冻结 BatchNorm 层（使用全局均值和方差，而不是当前 batch 的）。
    # 如果漏掉这一步，你在手机上推理出的结果将是完全随机和错误的。
    model.eval()

    print("2. 正在构建 Dummy Input (伪造输入)...")
    # 构建一个形状与实际输入完全一致的“空”张量 (Tensor)。
    # 这里的维度 [1, 3, 224, 224] 分别代表：
    # 1: Batch Size (手机端推理通常一次只处理一张图)
    # 3: Channels (RGB 三个颜色通道)
    # 224, 224: 图像的宽高 (MobileNetV2 标准输入尺寸)
    example_input = torch.rand(1, 3, 224, 224)

    print("3. 正在进行 JIT Tracing (计算图追踪)...")
    # 【Infra 必考点 2：TorchScript 转换】
    # PyTorch 默认是动态图（代码运行到哪才构建哪），这对调试友好，但对端侧部署极其不友好（太慢、依赖 Python 环境）。
    # torch.jit.trace 的作用是：让伪造的输入在模型里“跑”一遍，
    # PyTorch 会在底层记录下所有经过的算子（加减乘除、卷积等），从而生成一张静态的“计算图”（TorchScript）。
    # 这样模型就脱离了 Python 环境，可以在纯 C++ 环境中运行。
    traced_script_module = torch.jit.trace(model, example_input)

    print("4. 正在进行端侧专项优化 (Optimize for Mobile)...")
    # 【Infra 必考点 3：算子融合与内存优化】
    # 这一步是端侧性能优化的核心。它会在底层重写刚刚生成的计算图。
    # 最典型的操作是“算子融合 (Operator Fusion)”：
    # 比如它会将 [Conv2d卷积] -> [BatchNorm] -> [ReLU激活] 这三个原本独立的算子，
    # 在数学等价的前提下，强行合并成一个大算子。
    # 收益：大幅减少 CPU/内存之间的数据来回搬运 (Memory I/O)，降低推理延迟。
    optimized_traced_model = optimize_for_mobile(traced_script_module)

    print("5. 正在保存为 PyTorch Lite 格式...")
    # 将优化后的模型序列化并保存到磁盘。
    # _save_for_lite_interpreter 是专门为移动端 Lite 解释器设计的轻量级格式。
    # 相比标准的 .pt 文件，.ptl 去掉了大量移动端用不到的调试信息和元数据，体积更小，加载更快。
    output_path = "mobilenet_v2_optimized.ptl"
    optimized_traced_model._save_for_lite_interpreter(output_path)
    traced_script_module._save_for_lite_interpreter("mobilenet_v2_trace_only.ptl")

    print(f"✅ 模型导出成功！文件已保存在: {output_path}")


if __name__ == "__main__":
    export_to_mobile()
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import sys

try:
    # 加载模型
    model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()

    # 图片预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open('app/src/main/res/mipmap-xxhdpi/fish.png').convert('RGB')

    # 把没做 tensor 处理的中心点拿出来看下原色
    img_resized = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img_resized)
    print(f"Raw RGB at (112, 112): R={arr[112, 112, 0]}, G={arr[112, 112, 1]}, B={arr[112, 112, 2]}")

    input_tensor = transform(img).unsqueeze(0)

    # 打印一些验证信息
    print(f"Tensor Pixel of input_tensor at (112, 112): \n"
          f"R: {input_tensor[0, 0, 112, 112].item()}\n"
          f"G: {input_tensor[0, 1, 112, 112].item()}\n"
          f"B: {input_tensor[0, 2, 112, 112].item()}")

    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        top5_prob, top5_catid = torch.topk(output, 5)
        print(f"Top 5 Indices: {top5_catid[0].tolist()}")
        print(f"Top 5 Scores: {top5_prob[0].tolist()}")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

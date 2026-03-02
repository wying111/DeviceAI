import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
try:
    # 图片预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open('app/src/main/res/mipmap-xxhdpi/fish.png').convert('RGB')
    input_tensor = transform(img).unsqueeze(0)

    for model_path in ["mobilenet_v2_optimized.ptl", "mobilenet_v2_trace_only.ptl"]:
        print(f"Loading mobile optimized model: {model_path}")
        
        # 注意：.ptl 是一个 JIT module，可以直接通过 torch.jit.load 加载进 Python
        model = torch.jit.load(model_path)
        model.eval()

        # 推理
        with torch.no_grad():
            output = model(input_tensor)
            top5_prob, top5_catid = torch.topk(output, 5)
            print(f"Top 5 Indices for {model_path}: {top5_catid[0].tolist()}")
            print(f"Top 5 Scores for {model_path}: {top5_prob[0].tolist()}")
            print("-" * 30)

except Exception as e:
    import traceback
    traceback.print_exc()

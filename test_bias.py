import torch
import torchvision

model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
bias = model.classifier[1].bias

top5_prob, top5_catid = torch.topk(bias, 5)
print(f"Top 5 Bias Indices: {top5_catid.tolist()}")
print(f"Top 5 Bias Values: {top5_prob.tolist()}")

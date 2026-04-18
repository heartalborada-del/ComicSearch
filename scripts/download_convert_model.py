from pathlib import Path
import torch
import open_clip

output_dir = Path("../models")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. 加载模型
print("加载模型...")
#model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
#tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K')

model.eval()

# 2. 导出视觉编码器为 ONNX
print("导出视觉编码器...")
vision_model = model.visual
dummy_pixel_values = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    x = torch.randn(1, 3, 224, 224)
    y = model.encode_image(x)
    print(y.shape)  # 期望类似 [1, 768]

torch.onnx.export(
    vision_model,
    (dummy_pixel_values,),
    output_dir / "vision_model.onnx",
    input_names=["pixel_values"],
    output_names=["image_features"],
    opset_version=18,
    do_constant_folding=True,
)

print(f"ONNX 导出完成: {output_dir.resolve()}")
print(f"文件: {list(output_dir.glob('*.onnx'))}")

from ultralytics import YOLO

# 【请修改为您 best.pt 文件的实际完整路径】
model_path = r'C:\tuxiangxuexi\yachi\code\runs\segment\train21\weights\best.pt'
# 或者，如果您是从项目根目录运行这个测试脚本，也可以用相对路径：
# model_path = 'runs/segment/train9/weights/best.pt'

try:
    print(f"正在尝试加载模型: {model_path}")
    model = YOLO(model_path)
    print("模型加载成功！")
    print(f"模型类型: {type(model)}")
    # 您可以进一步打印模型的简要信息，比如它识别的类别名称 (如果已加载)
    if hasattr(model, 'names'):
        print(f"模型识别的类别: {model.names}")
    else:
        print("模型中未直接找到类别名称属性 (names)，这通常在加载数据或训练后设置。")

except Exception as e:
    print(f"加载模型失败！错误信息: {e}")
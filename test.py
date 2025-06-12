# --- 最小验证脚本 ,验证模型是否可用---
from ultralytics import YOLO

# --- 请修改下面这三行 ---
MODEL_PATH = r"runs/segment/train21/weights/best.pt"  # 1. 换成您自己训练好的模型的【绝对路径】
IMAGE_PATH = r"C:\tuxiangxuexi\yachi\code\datasets\data\train\images\4f65079a-MOHAMADI_HAMID_2020-07-19190541_jpg.rf.160e7975f8e9f5e990ad1bdbaa922886.jpg"          # 2. 换成您要分析的图片的【绝对路径】
CONF_THRESHOLD = 0.1                               # 3. 使用一个很低的置信度来测试

# --- 运行预测 ---
try:
    print(f"正在用模型 '{MODEL_PATH}' 预测图片 '{IMAGE_PATH}'...")
    model = YOLO(MODEL_PATH)
    results = model.predict(source=IMAGE_PATH, conf=CONF_THRESHOLD)

    # --- 打印结果 ---
    detected = False
    for r in results:
        if len(r.boxes) > 0:
            detected = True
            print("\n成功检测到目标！以下是详细信息：")
            print(r.boxes)  # 打印检测到的所有框的信息 (坐标, 置信度, 类别ID)
        
    if not detected:
        print("\n在此置信度下，模型未能检测到任何目标。")

except Exception as e:
    print(f"\n程序出错: {e}")

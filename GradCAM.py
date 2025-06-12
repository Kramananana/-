import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# -------------------------------------------------
# 0. 与YOLOv8行为一致的Letterbox预处理函数
# -------------------------------------------------
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    将图像调整大小并填充为指定尺寸，以保持宽高比。
    这模拟了YOLOv8的内部预处理。
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im

# -------------------------------------------------
# 1. 为 YOLOv8 适配的 GradCAM 类 (最终修复)
# -------------------------------------------------
class YOLOV8GradCAM:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.feature_maps = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_feature_maps)
        self.target_layer.register_backward_hook(self._save_gradients)

    def _save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()
        
    def __call__(self, x, class_idx, conf_threshold=0.25):
        """
        生成GradCAM热力图的核心逻辑。
        """
        x = x.to(self.device)
        
        # 确保梯度计算被启用
        with torch.enable_grad():
            # 前向传播，获取原始输出
            outputs = self.model(x)
        
        # 从模型输出中解析预测结果
        predictions = outputs[0] if isinstance(outputs, tuple) else outputs

        # ----- 关键修复逻辑开始：使用分数求和代替筛选，以保证梯度流 -----
        # 1. 直接通过索引获取所有预测框对于“目标类别”的置信度分数
        target_class_scores = predictions[0, 4 + class_idx, :]
        
        # 2. 直接将所有该类别的分数相加. 这种方法对计算图更友好
        target_score = target_class_scores.sum()

        # 3. 在求和之后再检查总分是否有效
        if target_score.item() <= 0:
            print(f"警告: 类别 {class_idx} 的总置信度为零或负数。无法生成热力图。")
            return None
        # ----- 关键修复逻辑结束 -----

        print(f"为类别 {class_idx} 计算的置信度总和为: {target_score.item():.4f}")

        # 从总分开始进行反向传播
        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        if self.gradients is None:
            raise RuntimeError("无法获取梯度。请检查目标层是否正确或模型结构。")

        # 使用梯度和特征图计算GradCAM
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)
        cam = torch.nn.functional.interpolate(cam, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        
        return cam.squeeze().cpu().numpy()

# -------------------------------------------------
# 2. 定义可视化函数 (保持不变)
# -------------------------------------------------
def visualize_yolo_gradcam(img_path, cam, results):
    img = cv2.imread(img_path)
    if cam is None:
        return img 
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    plot_img = results[0].plot() # 使用YOLO自带的绘图功能，包含框和标签
    final_img = cv2.addWeighted(plot_img, 0.6, heatmap, 0.4, 0)
    return final_img

# -------------------------------------------------
# 3. 主程序
# -------------------------------------------------
if __name__ == '__main__':
    # --- 用户需要修改的部分 ---
    MODEL_PATH = r"runs/segment/train21/weights/best.pt" # <--- 请务必替换为您自己训练好的模型路径!
    IMAGE_PATH = r"C:\tuxiangxuexi\yachi\code\datasets\data\train\images\4f65079a-MOHAMADI_HAMID_2020-07-19190541_jpg.rf.160e7975f8e9f5e990ad1bdbaa922886.jpg" # <--- 请替换为您想分析的图片
    TARGET_CLASS_INDEX = 0 # 分析 "Caries"
    CONF_THRESHOLD = 0.25 
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用的设备: {DEVICE}")

    # --- 程序开始 ---
    try:
        model = YOLO(MODEL_PATH)
        model.to(DEVICE)
        # 我们让模型保持在默认模式，不在此时调用 model.eval()
    except Exception as e:
        print(f"加载模型失败: {e}")
        exit()

    try:
        # 对于分割模型，最后一个C2f通常在 model.model[-2]
        target_layer = model.model.model[-2]
        print(f"选择的目标层: {target_layer.__class__.__name__}")
    except Exception as e:
        print(f"选择目标层失败: {e}")
        exit()
        
    try:
        img_bgr = cv2.imread(IMAGE_PATH)
        img_letterboxed = letterbox(img_bgr, (640, 640), auto=False)
        img_tensor = torch.from_numpy(np.transpose(img_letterboxed, (2, 0, 1))).float() / 255.0
        input_tensor = img_tensor.unsqueeze(0)
    except Exception as e:
        print(f"加载或预处理图片失败: {e}")
        exit()
        
    # 初始化并运行 GradCAM
    yolo_gradcam = YOLOV8GradCAM(model=model.model, target_layer=target_layer, device=DEVICE)
    cam_array = yolo_gradcam(input_tensor, class_idx=TARGET_CLASS_INDEX, conf_threshold=CONF_THRESHOLD)
    
    # 可视化结果并保存
    if cam_array is not None:
        # 关键修复：在GradCAM成功运行之后，再调用predict来获取用于可视化的检测框
        results = model.predict(IMAGE_PATH, conf=CONF_THRESHOLD, verbose=False)
        visualized_image = visualize_yolo_gradcam(IMAGE_PATH, cam_array, results)
        output_filename = "yolov8_gradcam_result_final.jpg"
        cv2.imwrite(output_filename, visualized_image)
        print(f"\n可视化结果已成功保存为文件: {output_filename}")
    else:
        print("\n由于未检测到置信度足够高的目标类别，无法生成GradCAM图像。")
        print("建议尝试：\n1. 更换一张模型能高置信度检测出目标的图片。\n2. 适当调低 CONF_THRESHOLD 的值。")

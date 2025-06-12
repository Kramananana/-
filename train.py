import os
import torch
import yaml
from ultralytics import YOLO  # 导入YOLO模型
from QtFusion.path import abs_path
import matplotlib
matplotlib.use('TkAgg')


if __name__ == '__main__':  # 确保该模块被直接运行时才执行以下代码
    workers = 1
    batch = 8  # 适当等修改Batchsize，根据电脑等显存/内存设置，如果爆显存可以调低
    device = 0 #if torch.cuda.is_available() else "cpu"

    data_path = abs_path(f'datasets/data/data.yaml', path_type='current')  # 数据集的yaml的绝对路径

    unix_style_path = data_path.replace(os.sep, '/')
    # 获取目录路径
    directory_path = os.path.dirname(unix_style_path)
    # 读取YAML文件，保持原有顺序
    with open(data_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    # 修改path项
    if 'train' in data and 'val' in data and 'test' in data:
        data['train'] = directory_path + '/train'
        data['val'] = directory_path + '/val'
        data['test'] = directory_path + '/test'

        # 将修改后的数据写回YAML文件
        with open(data_path, 'w') as file:
            yaml.safe_dump(data, file, sort_keys=False)
    # 注意！不同模型大小不同，对设备等要求不同，如果要求较高的模型【报错】则换其他模型测试即可（例如yolov8-seg.yaml、yolov8-seg-goldyolo.yaml、yolov8-seg-C2f-Faster.yaml、yolov8-seg-C2f-DiverseBranchBlock.yaml、yolov8-seg-efficientViT.yaml等）
    #model = YOLO(r"C:\codeseg\codenew\50+种YOLOv8算法改进源码大全和调试加载训练教程（非必要）\改进YOLOv8模型配置文件\yolov8-seg-C2f-Faster.yaml").load("./weights/yolov8s-seg.pt")
    #model = YOLO(model='./yolov8s-seg.yaml', task='segment').load('./weights/yolov8s-seg.pt')  # 加载预训练的YOLOv8模型
    #model = YOLO('runs/segment/train9/weights/best.pt')
    model = YOLO('yolov8s-seg.pt')  

    results = model.train(
        data=data_path,
        device=device,
        workers=workers,
        imgsz=640,
        epochs=300, # 或者您想尝试的新轮数
        batch=batch,
        # 开始训练模型
        # data=data_path,  # 指定训练数据的配置文件路径
        # device=device,  # 自动选择进行训练
        # workers=workers,  # 指定使用2个工作进程加载数据
        # imgsz=640,  # 指定输入图像的大小为640x640
        # epochs=200,  # 指定训练100个epoch
        # batch=batch,  # 指定每个批次的大小为8
    )

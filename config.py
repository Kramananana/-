import os
# --- 基本配置 ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SECRET_KEY = 'a_very_secret_key_for_flask_app_please_change_this'

# --- 数据库配置 ---
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'dental_app.db')
SQLALCHEMY_TRACK_MODIFICATIONS = False

# --- 文件夹路径 ---
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
CHAT_LOGS_FOLDER = os.path.join(BASE_DIR, 'chat_logs')

# --- 模型与API配置 ---
# DeepSeek API Key, 请替换为您自己的Key
DEEPSEEK_API_KEY = "sk-24d5d08e7dc341f994e8a6c3d61bb4ac" # 警告：请勿将API Key直接硬编码在生产环境中

# YOLOv8 模型文件路径
MODEL_PATH = os.path.join(BASE_DIR, 'runs/segment/train21/weights/best.pt') 

# --- 其他配置 ---
# 医生注册码
DOCTOR_REGISTRATION_CODE = "1234"

# 用于在图片上绘制中文的字体文件路径
# 请确保该路径在您的系统中是有效的，否则图片上的中文标签将无法显示
# 这是一个常见的Windows宋体路径，如果您的系统不同，请修改它
FONT_PATH = "C:/Windows/Fonts/simsun.ttc" 
if not os.path.exists(FONT_PATH):
    FONT_PATH = None # 如果字体不存在，则置为None，程序会进行回退处理
    print(f"警告: 找不到字体文件 '{FONT_PATH}'。AI识别结果中的中文标签可能无法正确显示。")
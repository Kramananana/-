import os
import json
import datetime
import numpy as np
import cv2
import openai
import pandas as pd
import io
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

import config
from models import db, User, Patient, XRay, Appointment, Prescription

# --- 全局变量与初始化 ---
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.CHAT_LOGS_FOLDER, exist_ok=True)

# 中英文类别名映射
Chinese_Name_Mapping = {"Caries": "龋齿", "Periapical lesion": "根尖周病变"}
CLASS_ID_TO_NAME = {0: "Caries", 1: "Periapical lesion"}
CLASS_COLORS = {"Caries": (255, 0, 0), "Periapical lesion": (0, 0, 255)} # 龋齿用红色，根尖周病变用蓝色

yolo_model = None
deepseek_client = None

def load_models():
    """在应用启动时加载YOLO模型和初始化AI客户端"""
    global yolo_model, deepseek_client
    if yolo_model is None:
        try:
            if os.path.exists(config.MODEL_PATH):
                yolo_model = YOLO(config.MODEL_PATH)
                print(f"YOLO模型加载成功: {config.MODEL_PATH}")
            else:
                print(f"错误: YOLO模型文件未找到: {config.MODEL_PATH}")
        except Exception as e:
            print(f"加载YOLO模型时发生严重错误: {e}")

    if deepseek_client is None and config.DEEPSEEK_API_KEY:
        try:
            deepseek_client = openai.OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
            print("DeepSeek AI客户端初始化成功！")
        except Exception as e:
            print(f"DeepSeek AI客户端初始化失败: {e}")

def init_database(app):
    """初始化数据库，如果为空则创建默认用户和数据"""
    with app.app_context():
        db.create_all()
        if not User.query.first():
            print("数据库为空，正在创建默认用户和示例数据...")
            # 创建用户
            admin = User(username='admin', role='admin')
            admin.set_password('admin')
            doctor = User(username='doctor1', role='doctor')
            doctor.set_password('doctor1')
            patient_user = User(username='patient1', role='patient')
            patient_user.set_password('patient1')
            db.session.add_all([admin, doctor, patient_user])
            db.session.commit()
            
            # 创建一条示例病历
            example_patient = Patient(
                name='patient1',
                gender='男',
                dob='1990-05-20',
                contact='13800138000',
                chief_complaint='右下后牙疼痛一周',
                present_illness='一周前开始感觉右下后牙遇冷热刺激疼痛，夜间加剧。',
                past_history='无特殊系统性疾病史。',
                examination_info='46牙合面见深龋洞，探（+），叩（++），冷测激发剧痛。',
                differential_diagnosis='1. 急性牙髓炎 2. 根尖周炎',
                treatment_plan='建议行根管治疗。',
                doctor_id=doctor.id
            )
            db.session.add(example_patient)
            db.session.commit()
            print("默认用户和示例病历创建成功。")

# --- 用户管理 ---
def get_user_by_username(username):
    return User.query.filter_by(username=username).first()

def add_user(username, password, role):
    if get_user_by_username(username):
        return False, "用户名已存在"
    new_user = User(username=username, role=role)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    return True, "注册成功"

def update_user(username, new_password=None, new_role=None):
    user = get_user_by_username(username)
    if user:
        if new_password:
            user.set_password(new_password)
        if new_role:
            user.role = new_role
        db.session.commit()
        return True
    return False

def delete_user(username):
    user = get_user_by_username(username)
    if user:
        # 删除用户时，其关联的患者、预约等会因模型中的cascade设置而被一并删除
        db.session.delete(user)
        db.session.commit()
        return True
    return False

def verify_user(username, password):
    user = get_user_by_username(username)
    return user if user and user.check_password(password) else None

# --- 患者管理 ---
def get_patient_by_id(patient_id):
    return Patient.query.get(patient_id)

def get_patient_by_name(patient_name):
    return Patient.query.filter_by(name=patient_name).first()

def add_patient(data):
    if get_patient_by_name(data['name']):
        return None, "患者姓名已存在，请使用其他名称。"
    
    doctor = get_user_by_username(data['doctor_username'])
    if not doctor:
        return None, "关联的医生不存在"

    new_patient = Patient(doctor_id=doctor.id)
    for k, v in data.items():
        if k != 'doctor_username' and hasattr(new_patient, k):
            setattr(new_patient, k, v)
    db.session.add(new_patient)
    db.session.commit()
    return new_patient, "新患者档案创建成功"

def update_patient(patient_id, data):
    patient = get_patient_by_id(patient_id)
    if patient:
        for key, value in data.items():
            if hasattr(patient, key):
                setattr(patient, key, value)
        db.session.commit()
        return True
    return False

def delete_patient(patient_id):
    patient = get_patient_by_id(patient_id)
    if patient:
        db.session.delete(patient)
        db.session.commit()
        return True
    return False

# --- X光片处理 ---
def add_xray_to_patient(patient_id, xray_data):
    """向患者档案中添加一条新的X光片记录"""
    patient = get_patient_by_id(patient_id)
    if patient:
        # 使用 **xray_data 可以优雅地将字典中的所有键值对作为参数传入
        new_xray = XRay(patient_id=patient.id, **xray_data)
        db.session.add(new_xray)
        db.session.commit()
        return True
    return False

def delete_xray_from_patient(xray_id):
    """根据ID删除X光片记录及其关联的两个文件"""
    xray = XRay.query.get(xray_id)
    if xray:
        try:
            # 1. 删除原始文件
            original_filepath = os.path.join(config.UPLOAD_FOLDER, xray.filename)
            if os.path.exists(original_filepath):
                os.remove(original_filepath)
            
            # 2. 【修复】根据原始文件名推断并删除AI标注图文件
            base, _ = os.path.splitext(xray.filename)
            overlay_filename = f"overlay_{base}.png"
            overlay_filepath = os.path.join(config.UPLOAD_FOLDER, overlay_filename)
            if os.path.exists(overlay_filepath):
                os.remove(overlay_filepath)

        except OSError as e:
            print(f"删除X光片文件时出错: {e}")
        
        # 3. 从数据库删除记录
        db.session.delete(xray)
        db.session.commit()
        return True
    return False

# --- AI 核心功能 ---
def run_yolo_inference(image_path):
    """使用YOLOv8对指定路径的图片进行AI推理"""
    if not yolo_model:
        print("YOLO 模型未加载，无法进行推理。")
        return []
    try:
        results = yolo_model.predict(source=image_path, save=False, verbose=False)
        
        detections = []
        if not results: return []
        result = results[0]
        
        if result.boxes is None or result.masks is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        masks = result.masks.data.cpu().numpy()
        
        for box, conf, class_id, mask in zip(boxes, confs, class_ids, masks):
            detections.append({
                'box': box.tolist(),
                'confidence': float(conf),
                'class_name': CLASS_ID_TO_NAME.get(class_id, "未知"),
                'mask': mask.tolist()
            })
        return detections
    except Exception as e:
        print(f"YOLO推理过程中发生错误: {e}")
        return []

def draw_overlays_on_image(img_pil, ai_results, show_confidence=True):
    """在PIL图片上绘制AI识别结果（分割区域、边界框、标签）"""
    if not ai_results:
        return img_pil.convert('RGB')

    img_cv = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
    overlay = img_cv.copy()
    
    try:
        font = ImageFont.truetype(config.FONT_PATH, 15) if config.FONT_PATH else ImageFont.load_default()
    except IOError:
        print(f"警告: 无法加载字体 {config.FONT_PATH}。将使用默认字体。")
        font = ImageFont.load_default()

    for res in ai_results:
        class_name = res['class_name']
        color = CLASS_COLORS.get(class_name, (0, 255, 0))
        
        if res.get('mask'):
            mask = np.array(res['mask'], dtype=np.uint8)
            mask_resized = cv2.resize(mask, (img_cv.shape[1], img_cv.shape[0]), interpolation=cv2.INTER_NEAREST)
            colored_mask = np.zeros_like(img_cv, dtype=np.uint8)
            colored_mask[mask_resized > 0] = color
            cv2.addWeighted(overlay, 1, colored_mask, 0.4, 0, overlay)

        x1, y1, x2, y2 = map(int, res['box'])
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
    
    final_image_cv = cv2.addWeighted(overlay, 0.6, img_cv, 0.4, 0)
    final_image_pil = Image.fromarray(cv2.cvtColor(final_image_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(final_image_pil)
    
    for res in ai_results:
        class_name = res['class_name']
        color = CLASS_COLORS.get(class_name, (0, 255, 0))
        x1, y1, _, _ = map(int, res['box'])
        
        label = Chinese_Name_Mapping.get(class_name, class_name)
        if show_confidence:
            label += f" {res['confidence']:.2f}"
            
        # 【修复】使用 font.getbbox() 来替代废弃的 textsize()
        try:
            # font.getbbox() 返回 (left, top, right, bottom)
            text_box = font.getbbox(label)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]
            
            # 定义文本背景框
            text_bg_rect = [x1, y1 - text_h - 4, x1 + text_w + 4, y1]
            
            # 绘制文本背景和文本
            draw.rectangle(text_bg_rect, fill=color)
            draw.text((x1 + 2, y1 - text_h - 2), label, fill=(255, 255, 255), font=font)

        except Exception as e:
            print(f"绘制文本时发生错误: {e}. 将只绘制无背景文本。")
            draw.text((x1, y1 - 15), label, fill=color, font=font)


    return final_image_pil

# --- 聊天功能 ---
def load_chat_history(username, role):
    """加载聊天记录，患者从数据库加载，其他角色从文件加载"""
    if role == 'patient':
        patient = get_patient_by_name(username)
        return patient.chat_history if patient and patient.chat_history else []
    
    log_file = os.path.join(config.CHAT_LOGS_FOLDER, f"{username}_chat.json")
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_chat_history(username, role, history):
    """保存聊天记录"""
    history_to_save = [msg for msg in history if not msg.get('is_context')]
    if role == 'patient':
        patient = get_patient_by_name(username)
        if patient:
            patient.chat_history = history_to_save
            db.session.commit()
        return
    
    log_file = os.path.join(config.CHAT_LOGS_FOLDER, f"{username}_chat.json")
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(history_to_save, f, indent=2, ensure_ascii=False)

def format_medical_record_for_ai(patient):
    """将患者的病历信息格式化为一段提供给大模型的上下文"""
    if not patient:
        return ""
    
    record = [f"你是一位专业的AI牙科医生，现在需要你根据以下患者的病历信息进行分析和回答。请使用友好、专业且易于理解的语言。"]
    record.append("\n--- 患者基本信息 ---")
    record.append(f"姓名: {patient.name}")
    if patient.gender: record.append(f"性别: {patient.gender}")
    if patient.dob: record.append(f"出生日期: {patient.dob}")
    if patient.contact: record.append(f"联系方式: {patient.contact}")
    
    record.append("\n--- 病历详情 ---")
    if patient.chief_complaint: record.append(f"主诉: {patient.chief_complaint}")
    if patient.present_illness: record.append(f"现病史: {patient.present_illness}")
    if patient.past_history: record.append(f"既往史: {patient.past_history}")
    if patient.examination_info: record.append(f"检查: {patient.examination_info}")
    if patient.differential_diagnosis: record.append(f"鉴别诊断: {patient.differential_diagnosis}")
    if patient.treatment_plan: record.append(f"治疗计划: {patient.treatment_plan}")

    if patient.xrays.all():
        record.append("\n--- X光片AI分析摘要 ---")
        for xray in patient.xrays:
            findings = [f"{Chinese_Name_Mapping.get(res['class_name'], res['class_name'])} (置信度: {res['confidence']:.2f})" for res in xray.ai_results]
            if findings:
                record.append(f"于 {xray.upload_date} 上传的X光片中发现: {', '.join(findings)}。")
            else:
                record.append(f"于 {xray.upload_date} 上传的X光片中未发现明显异常。")
    
    record.append("\n--- 指令 ---")
    record.append("请基于以上信息，回答用户接下来的问题。")
    
    return "\n".join(record)


def get_deepseek_response_stream(messages, model="deepseek-chat"):
    """从DeepSeek API获取流式响应"""
    if not deepseek_client:
        yield "错误：AI服务未配置，请检查API Key。"
        return

    # 过滤掉自定义的 is_context 字段
    api_messages = [{k: v for k, v in msg.items() if k != 'is_context'} for msg in messages]

    try:
        response_stream = deepseek_client.chat.completions.create(
            model=model,
            messages=api_messages,
            stream=True
        )
        for chunk in response_stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except openai.APIConnectionError as e:
        print(f"DeepSeek API 连接错误: {e}")
        yield "无法连接到AI服务，请检查网络连接。"
    except openai.RateLimitError as e:
        print(f"DeepSeek API 请求频率超限: {e}")
        yield "AI服务请求过于频繁，请稍后再试。"
    except openai.APIStatusError as e:
        print(f"DeepSeek API 状态错误 (HTTP Status: {e.status_code}): {e.response}")
        yield f"AI服务返回错误 (代码: {e.status_code})，请联系管理员。"
    except Exception as e:
        print(f"与DeepSeek通信时发生未知错误: {e}")
        yield "与AI服务通信时发生未知错误。"

# --- 预约管理 ---
def add_appointment(data):
    patient = get_patient_by_id(data['patient_id'])
    if not patient:
        return None, "找不到患者"
    
    try:
        appointment_time = datetime.datetime.fromisoformat(data.get('appointment_time'))
    except (ValueError, TypeError):
        return None, "预约时间格式无效，请重新选择。"

    new_appointment = Appointment(
        patient_id=patient.id,
        doctor_id=patient.doctor_id,
        appointment_time=appointment_time,
        reason=data.get('reason'),
        status='待确认'
    )
    db.session.add(new_appointment)
    db.session.commit()
    return new_appointment, "您的预约请求已成功发送给医生！"

def get_appointment_by_id(appointment_id):
    return Appointment.query.get(int(appointment_id))

def update_appointment_status(appointment_id, status):
    appointment = get_appointment_by_id(appointment_id)
    if appointment and status in ['已确认', '已完成', '已取消']:
        appointment.status = status
        db.session.commit()
        return True
    return False

# --- 处方管理 (新版) ---
def parse_medications_from_textarea(text):
    """从多行文本中解析药品信息"""
    medications = []
    lines = text.strip().split('\n')
    for line in lines:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        med = {'name': parts[0]}
        med['dosage'] = parts[1] if len(parts) > 1 else ''
        med['frequency'] = parts[2] if len(parts) > 2 else ''
        med['notes'] = parts[3] if len(parts) > 3 else ''
        medications.append(med)
    return medications

def format_medications_for_textarea(medications):
    """将药品列表格式化为多行文本"""
    if not medications:
        return ""
    lines = []
    for med in medications:
        # 使用 get 方法避免键不存在的错误
        parts = [
            med.get('name', ''),
            med.get('dosage', ''),
            med.get('frequency', ''),
            med.get('notes', '')
        ]
        # 移除末尾的空字符串，使格式更整洁
        while parts and parts[-1] == '':
            parts.pop()
        lines.append(', '.join(parts))
    return '\n'.join(lines)
    
def add_prescription(form_data, patient_id, doctor_username):
    patient = get_patient_by_id(patient_id)
    doctor = get_user_by_username(doctor_username)

    if not all([patient, doctor]):
        return None, "找不到患者或医生信息"
    
    medications_text = form_data.get('medications_text')
    medications = parse_medications_from_textarea(medications_text)

    if not medications:
        return None, "处方无效，至少需要包含一种药品。"

    new_prescription = Prescription(patient_id=patient.id, doctor_id=doctor.id, medications=medications)
    db.session.add(new_prescription)
    db.session.commit()
    return new_prescription, "电子处方已成功开具！"

def update_prescription(prescription_id, form_data):
    prescription = get_prescription_by_id(prescription_id)
    if not prescription:
        return None, "找不到该处方"
        
    medications_text = form_data.get('medications_text')
    medications = parse_medications_from_textarea(medications_text)

    if not medications:
        return None, "处方无效，至少需要包含一种药品。"
        
    prescription.medications = medications
    db.session.commit()
    return prescription, "处方已成功更新！"

def get_prescription_by_id(prescription_id):
    return Prescription.query.get(int(prescription_id))

def delete_prescription(prescription_id):
    prescription = get_prescription_by_id(prescription_id)
    if prescription:
        db.session.delete(prescription)
        db.session.commit()
        return True
    return False
    
# --- 数据导出 ---
def generate_patient_record_csv(patient):
    """将患者的完整病历信息生成为CSV格式的字符串"""
    if not patient:
        return None
    
    # 使用pandas创建数据结构
    data = {
        '项目': [
            '姓名', '性别', '出生日期', '联系方式', '主治医生', '主诉', '现病史', '既往史', '检查', '鉴别诊断', '治疗计划'
        ],
        '内容': [
            patient.name, patient.gender, patient.dob, patient.contact, patient.doctor.username,
            patient.chief_complaint, patient.present_illness, patient.past_history,
            patient.examination_info, patient.differential_diagnosis, patient.treatment_plan
        ]
    }
    df = pd.DataFrame(data)
    
    if patient.xrays.all():
        xray_df = pd.DataFrame(columns=['项目', '内容'])
        for i, xray in enumerate(patient.xrays):
            findings = [f"{Chinese_Name_Mapping.get(res['class_name'], res['class_name'])} ({res['confidence']:.0%})" for res in xray.ai_results]
            finding_str = ', '.join(findings) if findings else '未见异常'
            xray_df.loc[i] = [f"X光片记录 ({xray.upload_date})", f"AI分析结果: {finding_str}"]
        df = pd.concat([df, xray_df], ignore_index=True)

    if patient.prescriptions.all():
        presc_df = pd.DataFrame(columns=['项目', '内容'])
        for i, presc in enumerate(patient.prescriptions):
            med_list = [f"{med['name']} ({med['dosage']}, {med['frequency']})" for med in presc.medications]
            presc_df.loc[i] = [f"处方记录 ({presc.date_issued})", '; '.join(med_list)]
        df = pd.concat([df, presc_df], ignore_index=True)

    output = io.StringIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    
    return output.getvalue()

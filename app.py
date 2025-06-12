import streamlit as st
import os
import openai
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import json
import datetime
import bcrypt
import cv2

# --- 配置项 ---
# 医生注册权限码 (请务必更改为一个更复杂和安全的密码!)
DOCTOR_REGISTRATION_CODE = "1234"

# 从 Streamlit secrets 中获取 DeepSeek API Key
# 确保在 session_state 中初始化 api_key，以供后续使用
if "deepseek_api_key" not in st.session_state:
    try:
        st.session_state["deepseek_api_key"] = st.secrets["DEEPSEEK_API_KEY"]
        print("DeepSeek API Key 从 secrets.toml 加载成功。") # 调试信息
    except KeyError:
        st.error("DeepSeek API Key 未设置。请在 .streamlit/secrets.toml 中设置 DEEPSEEK_API_KEY。")
        st.session_state["deepseek_api_key"] = None # 设置为None以禁用AI功能
        print("DeepSeek API Key 未找到。") # 调试信息

# --- NEW: 使用 @st.cache_resource 来初始化 DeepSeek 客户端 ---
# 确保客户端只在 API Key 存在时被初始化一次并缓存
@st.cache_resource(show_spinner=False) # 不显示缓存的spinner
def initialize_deepseek_client(api_key):
    if not api_key:
        print("DeepSeek AI 客户端初始化失败：API Key 为空。") # 调试信息
        return None
    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1" # DeepSeek API 的基础URL
        )
        print("DeepSeek AI 客户端已成功初始化并缓存。") # 调试信息
        return client
    except Exception as e:
        print(f"DeepSeek AI 客户端初始化失败: {e}") # 调试信息
        st.error(f"DeepSeek AI 客户端初始化失败: {e}") # 显示给用户
        return None

# 在每次脚本运行时，尝试获取或初始化 DeepSeek 客户端（它会被缓存）
# 这确保了即使 secrets.toml 后来才准备好，客户端也能在下次运行时被初始化
if st.session_state["deepseek_api_key"] and "deepseek_client" not in st.session_state:
    st.session_state["deepseek_client"] = initialize_deepseek_client(st.session_state["deepseek_api_key"])
elif not st.session_state["deepseek_api_key"]:
    st.session_state["deepseek_client"] = None # 如果API Key为空，客户端也设为None
# --- 路径辅助函数 ---
def resource_path(relative_path):
    """获取资源文件的绝对路径，兼容PyInstaller打包"""
    base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)

# 模型路径 (请根据你的实际位置修改，如果best.pt在code/models/下，改为'models/best.pt')
# 我假设 best.pt 在 code/runs/segment/train21/weights/best.pt 路径
MODEL_PATH = resource_path('runs/segment/train21/weights/best.pt') 

# 数据存储目录
UPLOAD_FOLDER = resource_path('uploads')
USERS_CSV = resource_path('users_data/users.csv')
PATIENTS_JSON = resource_path('patients_data/patients.json')

# 确保所有必要目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(resource_path('users_data'), exist_ok=True)
os.makedirs(resource_path('patients_data'), exist_ok=True)

# --- 类别名称映射 ---
try:
    from chinese_name_list import Chinese_Name_Mapping
except ImportError:
    Chinese_Name_Mapping = {
        "Caries": "龋齿",
        "Periapical lesion": "牙周病",
    }
    st.warning("无法导入 chinese_name_list.py，使用默认类别映射。请检查文件是否存在。")

CLASS_ID_TO_NAME = {
    0: "Caries",
    1: "Periapical lesion"
}

# --- YOLO 模型加载 ---
@st.cache_resource
def load_yolo_model(path):
    if not os.path.exists(path):
        st.error(f"模型文件未找到: {path}。请检查路径。")
        return None
    try:
        model = YOLO(path)
        st.success(f"YOLO模型从 {path} 加载成功！")
        return model
    except Exception as e:
        st.error(f"加载YOLO模型失败: {e}")
        return None

yolo_model = load_yolo_model(MODEL_PATH)

# --- 用户认证和数据管理 ---
def load_users():
    if not os.path.exists(USERS_CSV) or os.path.getsize(USERS_CSV) == 0:
        with open(USERS_CSV, 'w', encoding='utf-8') as f:
            f.write("username,password_hash,role\n")
            doctor_pass_hash = bcrypt.hashpw('password'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            patient_pass_hash = bcrypt.hashpw('password'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            admin_pass_hash = bcrypt.hashpw('admin_password_123'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8') # 默认管理员密码
            f.write(f"doctor1,{doctor_pass_hash},doctor\n")
            f.write(f"patient1,{patient_pass_hash},patient\n")
            f.write(f"admin1,{admin_pass_hash},admin\n") # 默认管理员用户
        st.info("已创建默认用户：doctor1/password (医生), patient1/password (患者), admin1/admin_password_123 (管理员)。")
    users_df = pd.read_csv(USERS_CSV, encoding='utf-8')
    return users_df

def authenticate_user(username, password):
    users_df = load_users()
    user_data = users_df[users_df['username'] == username]
    if not user_data.empty:
        stored_hashed_password = user_data['password_hash'].iloc[0]
        if bcrypt.checkpw(password.encode('utf-8'), stored_hashed_password.encode('utf-8')):
            return user_data.iloc[0]['role']
    return None

# --- 用户管理辅助函数 (用于管理员界面) ---
def update_user_in_csv(username_to_update, new_password=None, new_role=None):
    users_df = load_users()
    user_index = users_df[users_df['username'] == username_to_update].index
    if not user_index.empty:
        if new_password:
            users_df.loc[user_index, 'password_hash'] = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        if new_role: 
            users_df.loc[user_index, 'role'] = new_role
        users_df.to_csv(USERS_CSV, index=False, encoding='utf-8')
        return True
    return False

def delete_user_from_csv(username_to_delete):
    users_df = load_users()
    original_count = len(users_df)
    users_df = users_df[users_df['username'] != username_to_delete]
    if len(users_df) < original_count:
        users_df.to_csv(USERS_CSV, index=False, encoding='utf-8')
        return True
    return False

# --- 患者数据管理功能 (使用JSON文件作为简易数据库) ---
def load_patients_data():
    if not os.path.exists(PATIENTS_JSON) or os.path.getsize(PATIENTS_JSON) == 0:
        with open(PATIENTS_JSON, 'w', encoding='utf-8') as f:
            json.dump({"patients": []}, f)
    with open(PATIENTS_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_patients_data(data):
    with open(PATIENTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_patient_by_id(patient_id):
    patients_data = load_patients_data()
    for patient in patients_data['patients']:
        if patient['id'] == patient_id:
            return patient
    return None
    
def get_patient_by_username(username):
    patients_data = load_patients_data()
    for patient in patients_data['patients']:
        if patient.get('name') == username: 
            return patient
    return None

def get_next_patient_id(patients_data):
    if not patients_data['patients']:
        return 1
    return max(p['id'] for p in patients_data['patients']) + 1

def get_next_xray_id(patient_data):
    if not patient_data.get('xrays'):
        return 1
    return max(xr['id'] for xr in patient_data['xrays']) + 1

# --- AI 识别功能 ---
def run_yolo_inference(image_path):
    if not yolo_model:
        st.error("AI模型未加载，无法进行识别。")
        return None
    try:
        results_list = yolo_model(image_path)
        parsed_results = []
        for r in results_list:
            if r.boxes is not None:
                for i, box in enumerate(r.boxes):
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    mask_data = None
                    if r.masks is not None and i < len(r.masks):
                        current_mask = r.masks[i]
                        if hasattr(current_mask, 'xy'):
                            segments = current_mask.xy[0].tolist()
                            mask_data = segments
                    
                    if mask_data:
                        parsed_results.append({
                            'type': 'mask',
                            'class_id': class_id,
                            'class_name': CLASS_ID_TO_NAME.get(class_id, f"Unknown_{class_id}"),
                            'confidence': conf,
                            'segments': [[float(p) for p in seg_point] for seg_point in mask_data]
                        })
                    else:
                        parsed_results.append({
                            'type': 'box',
                            'class_id': class_id,
                            'class_name': CLASS_ID_TO_NAME.get(class_id, f"Unknown_{class_id}"),
                            'confidence': conf,
                            'bbox': [float(b) for b in bbox]
                        })
        return parsed_results
    except Exception as e:
        st.error(f"AI识别时发生错误: {e}")
        st.exception(e)
        return None

# --- 辅助函数：在图像上绘制识别结果 ---
def display_image_with_overlays(img_pil, ai_results, show_confidence=True):
    img_np = np.array(img_pil.convert("RGB"))
    img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    try:
        font_path = "C:/Windows/Fonts/simsun.ttc"
        font = ImageFont.truetype(font_path, 25)
    except IOError:
        st.warning(f"无法加载中文字体：{font_path}。请检查路径或使用其他字体。")
        font = ImageFont.load_default()

    img_pil_draw = img_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img_pil_draw)
    
    COLOR_CARIES_RGB = (255, 0, 0) # 红色
    COLOR_PERIAPICAL_RGB = (0, 0, 255) # 蓝色
    COLOR_UNKNOWN_RGB = (255, 255, 0) # 黄色

    for item in ai_results:
        class_name_english = item.get('class_name')
        class_name_translated = Chinese_Name_Mapping.get(class_name_english, class_name_english)
        confidence = item.get('confidence')

        current_color_rgb = (0, 0, 0)
        if class_name_english == "Caries":
            current_color_rgb = COLOR_CARIES_RGB
        elif class_name_english == "Periapical lesion":
            current_color_rgb = COLOR_PERIAPICAL_RGB
        else:
            current_color_rgb = COLOR_UNKNOWN_RGB

        text = f"{class_name_translated}"
        if show_confidence:
            text += f" ({confidence:.2f})"
        
        if item['type'] == 'box':
            x1, y1, x2, y2 = item['bbox']
            draw.rectangle([x1, y1, x2, y2], outline=current_color_rgb, width=3)
            
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = x1
            text_y = y1 - text_height - 5
            if text_y < 0:
                text_y = y1 + 5
            
            draw.rectangle([text_x, text_y, text_x + text_width + 5, text_y + text_height + 5], fill=current_color_rgb)
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))

        elif item['type'] == 'mask':
            segments = np.array(item['segments'], dtype=np.int32)
            if segments.size > 0:
                flat_segments = segments.flatten().tolist()
                draw.polygon(flat_segments, fill=current_color_rgb + (int(255 * 0.4),))
                draw.line(flat_segments + [flat_segments[0], flat_segments[1]], fill=current_color_rgb, width=3)

                first_point_x, first_point_y = segments[0][0], segments[0][1]
                text_bbox = draw.textbbox((first_point_x, first_point_y), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = first_point_x
                text_y = first_point_y - text_height - 5
                if text_y < 0:
                    text_y = first_point_y + 5

                draw.rectangle([text_x, text_y, text_x + text_width + 5, text_y + text_height + 5], fill=current_color_rgb)
                draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))
    
    st.image(img_pil_draw) # REMOVED use_container_width

# --- 辅助函数：生成患者病历CSV ---
def generate_patient_record_csv(patient_data):
    if not patient_data:
        return ""

    record_data = {
        "患者ID": patient_data['id'],
        "姓名": patient_data['name'],
        "性别": patient_data['gender'],
        "出生日期": patient_data['dob'],
        "联系方式": patient_data['contact'],
        "关联医生": patient_data.get('doctor_username', 'N/A'),
        "主诉": patient_data.get('chief_complaint', 'N/A'),
        "现病史": patient_data.get('present_illness', 'N/A'),
        "既往史": patient_data.get('past_history', 'N/A'),
        "检查信息": patient_data.get('examination_info', 'N/A'),
        "鉴别诊断": patient_data.get('differential_diagnosis', 'N/A'),
        "治疗计划": patient_data.get('treatment_plan', 'N/A'),
    }

    xray_summary = []
    if patient_data.get('xrays'):
        for xray in patient_data['xrays']:
            xray_info = f"文件: {xray['filename']} (上传日期: {xray['upload_date']})"
            ai_counts = {}
            if xray.get('ai_results'):
                for item in xray['ai_results']:
                    if isinstance(item, dict) and 'class_name' in item:
                        class_name_english = item.get('class_name')
                        class_name_translated = Chinese_Name_Mapping.get(class_name_english, class_name_english)
                        ai_counts[class_name_translated] = ai_counts.get(class_name_translated, 0) + 1
            if ai_counts:
                ai_info_str = ", ".join([f"{name}: {count}" for name, count in ai_counts.items()])
                xray_info += f" [识别结果: {ai_info_str}]"
            else:
                xray_info += " [无识别结果]"
            xray_summary.append(xray_info)
    
    record_data["X光片历史摘要"] = "\n".join(xray_summary)

    df = pd.DataFrame([record_data])
    csv_buffer = df.to_csv(index=False, encoding='utf-8-sig')
    return csv_buffer

# --- [新增] 辅助函数: 为AI格式化病历信息 ---
def format_medical_record_for_ai(patient_data):
    """将患者数据格式化为一段清晰的文本，用作AI的上下文。"""
    if not patient_data:
        return ""

    # 提取最新的X光片信息
    latest_xray_summary = "无"
    if patient_data.get('xrays'):
        latest_xray = sorted(patient_data['xrays'], key=lambda x: x['upload_date'], reverse=True)[0]
        ai_counts = {}
        if latest_xray.get('ai_results'):
            for item in latest_xray['ai_results']:
                 if isinstance(item, dict) and 'class_name' in item:
                    class_name_english = item.get('class_name')
                    class_name_translated = Chinese_Name_Mapping.get(class_name_english, class_name_english)
                    ai_counts[class_name_translated] = ai_counts.get(class_name_translated, 0) + 1
        if ai_counts:
            latest_xray_summary = ", ".join([f"{name}: {count}处" for name, count in ai_counts.items()])
    
    # 构建文本摘要
    record_summary = f"""
### 患者病历摘要
这是关于当前与你对话的患者的背景信息。请在回答时参考这些信息，以提供更具个性化的建议。

- **姓名**: {patient_data.get('name', '未记录')}
- **性别**: {patient_data.get('gender', '未记录')}
- **主诉**: {patient_data.get('chief_complaint', '无')}
- **现病史**: {patient_data.get('present_illness', '无')}
- **既往史**: {patient_data.get('past_history', '无')}
- **检查信息**: {patient_data.get('examination_info', '无')}
- **治疗计划**: {patient_data.get('treatment_plan', '无')}
- **最新X光片AI识别摘要**: {latest_xray_summary}
---
"""
    return record_summary

# --- DeepSeek AI 对话功能 ---
def get_deepseek_response(messages, model="deepseek-chat"):
    """
    调用 DeepSeek API 获取聊天回复。
    messages: 聊天历史列表，格式为 [{"role": "user", "content": "hello"}, ...]
    model: 使用的DeepSeek模型名称
    """
    # 直接使用 session_state 中缓存的 DeepSeek 客户端
    client = st.session_state.get("deepseek_client") 
    
    if client is None: # 如果客户端仍为None，说明初始化失败或Key不存在
        st.error("DeepSeek AI 客户端未初始化，无法进行对话。请联系管理员或检查 API Key。")
        return "对不起，AI智能医生目前无法使用。"

    try:
        response = client.chat.completions.create( # 直接使用 client 变量
            model=model,
            messages=messages,
            stream=False # 非流式回复，一次性返回
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"DeepSeek AI 对话请求失败: {e}")
        st.exception(e)
        return "对不起，AI智能医生遇到问题，请稍后再试。"

# --- [修改] 辅助函数：AI 对话界面 ---
def chat_interface_page(user_role, patient_data=None):
    """
    为医生和患者提供一个与AI对话的界面。
    user_role: 当前用户的角色，用于区分聊天历史。
    patient_data: (可选) 当前患者的完整病历数据。
    """
    st.header(f"{user_role} AI 智能医生")
    
    st.warning("请注意：我是一个AI智能医生，无法提供专业医疗诊断或替代医生医嘱。所有对话内容仅供参考和科普，请务必咨询专业医生以获取准确的医疗指导和治疗方案。")
    
    # 根据用户角色显示不同的初始问候语
    if user_role == "患者":
        st.write("您好！我是您的AI智能医生，有什么可以帮助您的吗？")
    elif user_role == "医生":
        st.write("您好，医生！我是您的AI智能医生，能为您提供哪些信息或协助？")

    # 构建基础系统指令
    base_system_prompt = """你是一位专业的牙科医生AI智能医生。
你拥有丰富的牙科知识，擅长解答关于口腔健康、牙齿疾病（如龋齿、牙周炎）、牙齿护理、口腔卫生、牙科检查X光片解读（仅限科普性解释，不提供诊断）、常见牙科手术和治疗方案等问题。
请以专业、严谨、耐心且易懂的语言回答用户的问题。
**请务必强调你是一个AI智能医生，不能替代真正的医生进行诊断和治疗，所有建议仅供参考，用户应寻求专业牙医的面对面诊治。** 避免使用过于口语化的表达，保持医疗专业性。"""

    # 为患者生成个性化的系统指令
    if user_role == "患者" and patient_data:
        patient_context = format_medical_record_for_ai(patient_data)
        system_prompt = patient_context + base_system_prompt
    else:
        system_prompt = base_system_prompt
        
    # 使用 Streamlit Session State 来存储聊天历史
    # 每个用户的角色都有独立的聊天历史，以避免混淆
    session_key = f'chat_history_{user_role}'
    if session_key not in st.session_state:
        st.session_state[session_key] = []
        # 将我们构建好的系统指令作为第一条消息添加
        st.session_state[session_key].append({"role": "system", "content": system_prompt})
        print(f"DEBUG: 初始化 {user_role} 聊天历史，并添加了系统消息。")

    # 显示过去的聊天消息
    # 注意：在显示聊天历史时，跳过系统消息，因为它只是给AI看的，不应显示给用户
    for message in st.session_state[session_key]:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 用户的输入框
    prompt = st.chat_input("在这里输入您的问题...")

    if prompt: 
        st.session_state[session_key].append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("AI 正在思考..."):
                response = get_deepseek_response(st.session_state[session_key])
                st.markdown(response)
        
        st.session_state[session_key].append({"role": "assistant", "content": response}) 
        
    if st.button("清除聊天记录"):
        # 清除时不仅清空，还要重新添加系统消息，以便下次对话仍有角色设定
        st.session_state[session_key] = []
        st.session_state[session_key].append({"role": "system", "content": system_prompt})
        st.experimental_rerun() 
        print(f"DEBUG: {user_role} 聊天记录已清除，并重置了AI角色。")


# --- 编辑患者档案 ---
def patient_edit_form(selected_patient, doctor_username):
    st.subheader(f"编辑患者档案: {selected_patient['name']}")
    
    session_key_for_reset = f'edit_form_reset_key_{selected_patient["id"]}'
    if session_key_for_reset not in st.session_state:
        st.session_state[session_key_for_reset] = 0

    form_key = f"edit_patient_form_{selected_patient['id']}_{st.session_state[session_key_for_reset]}"
    with st.form(key=form_key):
        st.markdown("##### 基本信息")
        edited_name = st.text_input("姓名 *", value=selected_patient['name'], key=f"edit_name_{selected_patient['id']}")
        
        gender_options = ["男", "女", "其他"]
        current_gender_index = gender_options.index(selected_patient['gender']) if selected_patient['gender'] in gender_options else 0
        edited_gender = st.selectbox("性别", gender_options, index=current_gender_index, key=f"edit_gender_{selected_patient['id']}")
        
        edited_dob_str = selected_patient['dob']
        try:
            edited_dob = datetime.datetime.strptime(edited_dob_str, '%Y-%m-%d').date()
        except (ValueError, TypeError):
            edited_dob = datetime.date(2000, 1, 1)
            st.warning("出生日期格式异常，已设为默认值。")
        edited_dob = st.date_input("出生日期", value=edited_dob, key=f"edit_dob_{selected_patient['id']}")
        
        edited_contact = st.text_input("联系方式", value=selected_patient['contact'], key=f"edit_contact_{selected_patient['id']}")

        st.markdown("##### 病历信息")
        edited_chief_complaint = st.text_area("主诉", value=selected_patient.get('chief_complaint', ''), key=f"edit_chief_complaint_{selected_patient['id']}")
        edited_present_illness = st.text_area("现病史", value=selected_patient.get('present_illness', ''), key=f"edit_present_illness_{selected_patient['id']}")
        edited_past_history = st.text_area("既往史", value=selected_patient.get('past_history', ''), key=f"edit_past_history_{selected_patient['id']}")
        edited_examination_info = st.text_area("检查信息", value=selected_patient.get('examination_info', ''), key=f"edit_examination_info_{selected_patient['id']}")
        edited_differential_diagnosis = st.text_area("鉴别诊断", value=selected_patient.get('differential_diagnosis', ''), key=f"edit_differential_diagnosis_{selected_patient['id']}")
        edited_treatment_plan = st.text_area("治疗计划", value=selected_patient.get('treatment_plan', ''), key=f"edit_treatment_plan_{selected_patient['id']}")

        col_save, col_cancel = st.columns(2)
        with col_save:
            save_submitted = st.form_submit_button("保存修改") 
        with col_cancel:
            cancel_submitted = st.form_submit_button("取消修改") 

    if save_submitted:
        if not edited_name:
            st.error("姓名不能为空。")
        else:
            patients_data = load_patients_data()
            for i, p in enumerate(patients_data['patients']):
                if p['id'] == selected_patient['id']:
                    patients_data['patients'][i].update({
                        'name': edited_name,
                        'gender': edited_gender,
                        'dob': edited_dob.strftime('%Y-%m-%d'),
                        'contact': edited_contact,
                        'chief_complaint': edited_chief_complaint,
                        'present_illness': edited_present_illness,
                        'past_history': edited_past_history,
                        'examination_info': edited_examination_info,
                        'differential_diagnosis': edited_differential_diagnosis,
                        'treatment_plan': edited_treatment_plan,
                        'doctor_username': doctor_username
                    })
                    break
            save_patients_data(patients_data)
            st.success(f"患者 {edited_name} 档案已更新！")
            st.session_state[f'edit_mode_{selected_patient["id"]}'] = False
            st.session_state[session_key_for_reset] += 1 
            st.experimental_rerun()

    if cancel_submitted:
        st.info("修改已取消。")
        st.session_state[f'edit_mode_{selected_patient["id"]}'] = False
        st.session_state[session_key_for_reset] += 1
        st.experimental_rerun()

# --- Streamlit UI 界面函数 ---
def login_page():
    st.sidebar.image(resource_path('ini_image.png')) # REMOVED use_container_width
    st.sidebar.title("用户认证")

    login_tab, register_tab = st.tabs(["登录", "注册"])

    with login_tab:
        st.header("现有用户登录")
        username_login = st.text_input("用户名", key="username_login_input")
        password_login = st.text_input("密码", type="password", key="password_login_input")
        
        if st.button("登录", key="login_button"):
            role = authenticate_user(username_login, password_login)
            if role:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username_login
                st.session_state['role'] = role
                st.success("登录成功！")
                st.experimental_rerun()
            else:
                st.error("用户名或密码不正确。")
        st.info("默认用户：doctor1/password (医生), patient1/password (患者)")

    with register_tab:
        st.header("新用户注册")
        new_username = st.text_input("选择用户名 *", key="new_username_register_input")
        new_password = st.text_input("设置密码 *", type="password", key="new_password_register_input")
        confirm_password = st.text_input("确认密码 *", type="password", key="confirm_password_register_input")
        
        new_role = st.selectbox("选择角色", ["patient", "doctor"], key="new_role_register_select")

        registration_code = ""
        if new_role == "doctor":
            registration_code = st.text_input("医生权限码 *", type="password", key="doctor_registration_code_input")

        if st.button("注册", key="register_button"):
            users_df = load_users()
            if new_username in users_df['username'].values:
                st.error("该用户名已被占用，请选择其他用户名。")
            elif not new_username or not new_password or not confirm_password:
                st.error("用户名、密码和确认密码均不能为空。")
            elif new_password != confirm_password:
                st.error("两次输入的密码不一致，请重新输入。")
            elif new_role == "doctor" and registration_code != DOCTOR_REGISTRATION_CODE:
                st.error("医生权限码不正确。请联系管理员获取正确的权限码。")
            else:
                hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                
                new_user_df = pd.DataFrame([{
                    'username': new_username,
                    'password_hash': hashed_password,
                    'role': new_role
                }])
                
                new_user_df.to_csv(USERS_CSV, mode='a', header=False, index=False, encoding='utf-8')
                
                st.success(f"用户 '{new_username}' 注册成功！您现在可以登录了。")
    
    st.markdown("---")
    st.write("关于：这是一个结合YOLOv8模型进行识别与牙科影像管理的系统。")

def logout_button():
    if st.sidebar.button("登出"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['role'] = None
        st.info("您已成功登出。")
        st.experimental_rerun()

def doctor_dashboard_page():
    st.sidebar.image(resource_path('ini_image.png')) # REMOVED use_container_width
    st.sidebar.title(f"医生 {st.session_state['username']}")
    logout_button()

    if 'add_patient_message' not in st.session_state:
        st.session_state['add_patient_message'] = None
    if 'add_patient_form_key' not in st.session_state:
        st.session_state['add_patient_form_key'] = 0

    st.title("医生端 - 患者病历及影像信息管理")
    
    selected_tab = st.sidebar.radio("导航", ["添加患者", "查看/管理患者影像", "牙片图像识别系统"])

    if selected_tab == "添加患者":
        st.header("添加新患者病历档案")

        if st.session_state['add_patient_message']:
            if "成功" in st.session_state['add_patient_message']:
                st.success(st.session_state['add_patient_message'])
            else:
                st.warning(st.session_state['add_patient_message'])
            st.session_state['add_patient_message'] = None
        
        with st.form(key=f"add_patient_form_{st.session_state['add_patient_form_key']}"):
            st.subheader("基本信息")
            new_patient_name = st.text_input("患者姓名 *", key=f"new_patient_name_input_add_{st.session_state['add_patient_form_key']}")
            new_patient_gender = st.selectbox("性别", ["男", "女", "其他"], key=f"new_patient_gender_select_add_{st.session_state['add_patient_form_key']}")
            new_patient_dob = st.date_input("出生日期", datetime.date(2000, 1, 1), key=f"new_patient_dob_input_add_{st.session_state['add_patient_form_key']}")
            new_patient_contact = st.text_input("联系方式", key=f"new_patient_contact_input_add_{st.session_state['add_patient_form_key']}")

            st.subheader("病历信息")
            new_patient_chief_complaint = st.text_area("主诉", key=f"new_patient_chief_complaint_input_add_{st.session_state['add_patient_form_key']}")
            new_patient_present_illness = st.text_area("现病史", key=f"new_patient_present_illness_input_add_{st.session_state['add_patient_form_key']}")
            new_patient_past_history = st.text_area("既往史", key=f"new_patient_past_history_input_add_{st.session_state['add_patient_form_key']}")
            new_patient_examination_info = st.text_area("检查信息", key=f"new_patient_examination_info_input_add_{st.session_state['add_patient_form_key']}")
            new_patient_differential_diagnosis = st.text_area("鉴别诊断", key=f"new_patient_differential_diagnosis_input_add_{st.session_state['add_patient_form_key']}")
            new_patient_treatment_plan = st.text_area("治疗计划", key=f"new_patient_treatment_plan_input_add_{st.session_state['add_patient_form_key']}")

            submitted = st.form_submit_button("添加患者")

            if submitted:
                if new_patient_name:
                    patients_data = load_patients_data()
                    if any(p['name'] == new_patient_name for p in patients_data['patients']):
                        st.session_state['add_patient_message'] = f"患者 '{new_patient_name}' 已存在。请检查或更改姓名。"
                        st.session_state['add_patient_form_key'] += 1
                        st.experimental_rerun()
                        return
                    
                    new_patient_id = get_next_patient_id(patients_data)
                    patients_data['patients'].append({
                        'id': new_patient_id,
                        'name': new_patient_name,
                        'gender': new_patient_gender,
                        'dob': new_patient_dob.strftime('%Y-%m-%d'),
                        'contact': new_patient_contact,
                        'doctor_username': st.session_state['username'],
                        'chief_complaint': new_patient_chief_complaint,
                        'present_illness': new_patient_present_illness,
                        'past_history': new_patient_past_history,
                        'examination_info': new_patient_examination_info,
                        'differential_diagnosis': new_patient_differential_diagnosis,
                        'treatment_plan': new_patient_treatment_plan,
                        'xrays': []
                    })
                    save_patients_data(patients_data)
                    
                    st.session_state['add_patient_message'] = f"患者 {new_patient_name} 档案添加成功！"
                    
                    st.session_state['add_patient_form_key'] += 1
                    st.experimental_rerun()
                else:
                    st.session_state['add_patient_message'] = "患者姓名不能为空。"
                    st.session_state['add_patient_form_key'] += 1
                    st.experimental_rerun()

    elif selected_tab == "查看/管理患者影像":
        st.header("患者中心")
        patients_data = load_patients_data()
        my_patients = [p for p in patients_data['patients'] if p.get('doctor_username') == st.session_state['username']]
        
        if not my_patients:
            st.info("暂无患者档案。请前往 '添加患者' 标签页添加。")
            return

        search_query = st.text_input("搜索患者 (按姓名或联系方式)", key="patient_search_query")
        
        filtered_patients = []
        if search_query:
            search_query_lower = search_query.lower()
            for p in my_patients:
                if search_query_lower in p['name'].lower() or \
                   search_query_lower in p.get('contact', '').lower():
                    filtered_patients.append(p)
        else:
            filtered_patients = my_patients

        if not filtered_patients:
            st.info("未找到匹配的患者。")
            return

        patient_options = {p['name']: p['id'] for p in filtered_patients}
        
        if not list(patient_options.keys()):
            st.info("请添加患者或调整搜索条件。")
            st.session_state['current_selected_patient_id'] = None
            return
            
        current_selected_patient_name = st.selectbox("选择患者", list(patient_options.keys()), key="select_patient_for_xray")
        
        if st.session_state.get('current_selected_patient_id') != patient_options.get(current_selected_patient_name):
            st.session_state['current_selected_patient_id'] = patient_options.get(current_selected_patient_name)
            if st.session_state.get(f'edit_mode_{st.session_state["current_selected_patient_id"]}'):
                st.session_state[f'edit_mode_{st.session_state["current_selected_patient_id"]}'] = False
                st.experimental_rerun()

        if current_selected_patient_name:
            selected_patient_id = patient_options[current_selected_patient_name]
            selected_patient = get_patient_by_id(selected_patient_id)

            if selected_patient:
                edit_mode_key = f'edit_mode_{selected_patient["id"]}'
                if edit_mode_key not in st.session_state:
                    st.session_state[edit_mode_key] = False

                if st.session_state[edit_mode_key]:
                    patient_edit_form(selected_patient, st.session_state['username'])
                else:
                    st.subheader(f"患者信息: {selected_patient['name']}")
                    
                    if st.button("编辑病历信息", key=f"edit_patient_button_{selected_patient['id']}"):
                        st.session_state[edit_mode_key] = True
                        st.experimental_rerun()

                    st.write(f"**姓名:** {selected_patient['name']}")
                    st.write(f"**性别:** {selected_patient['gender']}")
                    st.write(f"**出生日期:** {selected_patient['dob']}")
                    st.write(f"**联系方式:** {selected_patient['contact']}")
                    
                    st.markdown("---")
                    st.subheader("病历详情")
                    st.write(f"**主诉:** {selected_patient.get('chief_complaint', 'N/A')}")
                    st.write(f"**现病史:** {selected_patient.get('present_illness', 'N/A')}")
                    st.write(f"**既往史:** {selected_patient.get('past_history', 'N/A')}")
                    st.write(f"**检查信息:** {selected_patient.get('examination_info', 'N/A')}")
                    st.write(f"**鉴别诊断:** {selected_patient.get('differential_diagnosis', 'N/A')}")
                    st.write(f"**治疗计划:** {selected_patient.get('treatment_plan', 'N/A')}")
                    st.markdown("---")

                    if st.button("导出病历 (CSV)", key=f"export_patient_record_csv_button_{selected_patient['id']}"):
                        csv_data = generate_patient_record_csv(selected_patient)
                        if csv_data:
                            export_filename = f"{selected_patient['name']}_病历_{datetime.date.today().strftime('%Y%m%d')}.csv"
                            st.download_button(
                                label="点击下载病历CSV",
                                data=csv_data,
                                file_name=export_filename,
                                mime="text/csv",
                                key=f"download_csv_button_{selected_patient['id']}"
                            )
                            st.success("病历CSV文件已生成，请点击下载按钮。")
                        else:
                            st.warning("未能生成病历CSV数据。")

                    st.subheader("上传新X光片")
                    if 'upload_xray_reset_key' not in st.session_state:
                        st.session_state['upload_xray_reset_key'] = 0

                    uploaded_file = st.file_uploader(
                        f"为 {selected_patient['name']} 上传X光片", 
                        type=['png', 'jpg', 'jpeg'], 
                        key=f"upload_xray_file_{st.session_state['upload_xray_reset_key']}"
                    )
                    
                    if uploaded_file is not None:
                        if 'last_uploaded_filename' not in st.session_state or st.session_state['last_uploaded_filename'] != uploaded_file.name:
                            st.session_state['last_uploaded_filename'] = uploaded_file.name

                            with st.spinner('正在上传并进行AI识别...'):
                                unique_filename = f"{selected_patient['id']}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{uploaded_file.name}"
                                file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())

                                ai_results = run_yolo_inference(file_path)
                                
                                patients_data_for_save = load_patients_data()
                                for i, p in enumerate(patients_data_for_save['patients']):
                                    if p['id'] == selected_patient['id']:
                                        new_xray_id = get_next_xray_id(p)
                                        patients_data_for_save['patients'][i]['xrays'].append({
                                            'id': new_xray_id,
                                            'filename': unique_filename,
                                            'upload_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                            'ai_results': ai_results # 会存储 None (错误) 或 [] (无检测) 或 [...] (有检测)
                                        })
                                        break
                                save_patients_data(patients_data_for_save)
                                
                                if ai_results is None:
                                    st.error("X光片上传成功，但AI识别过程中出现错误，请检查日志。")
                                elif not ai_results:
                                    st.info("X光片上传成功，未识别到病变。")
                                else:
                                    st.success("X光片上传并AI识别成功！")
                                
                                st.session_state['upload_xray_reset_key'] += 1
                                if 'last_uploaded_filename' in st.session_state:
                                    del st.session_state['last_uploaded_filename']
                                st.experimental_rerun()
                        else:
                            pass
                    
                    st.subheader("历史X光片")
                    if selected_patient.get('xrays'):
                        sorted_xrays = sorted(selected_patient['xrays'], key=lambda x: x['upload_date'], reverse=True)
                        
                        if 'xray_to_delete_confirm_id' not in st.session_state:
                            st.session_state['xray_to_delete_confirm_id'] = None
                        
                        for xray in sorted_xrays:
                            col_expander_title, col_delete_btn_container = st.columns([0.9, 0.1])
                            
                            with col_delete_btn_container:
                                delete_button_key = f"delete_xray_{xray['id']}_button"
                                if st.button("🗑️", key=delete_button_key, help="删除此X光片"):
                                    st.session_state['xray_to_delete_confirm_id'] = xray['id']
                                    st.experimental_rerun()

                            with col_expander_title:
                                # Removed key parameter for expander
                                with st.expander(f"X光片: {xray['filename']} (上传日期: {xray['upload_date']})", expanded=False):
                                    image_path = os.path.join(UPLOAD_FOLDER, xray['filename'])
                                    if os.path.exists(image_path):
                                        try:
                                            img_original = Image.open(image_path)
                                            
                                            st.markdown("##### 原始X光片")
                                            st.image(img_original, caption=f"原始X光片: {xray['filename']}") # REMOVED use_container_width

                                            if xray.get('ai_results'):
                                                st.markdown("##### AI识别结果")
                                                display_image_with_overlays(img_original.copy(), xray['ai_results'])

                                                st.subheader("识别统计:")
                                                current_xray_ai_results = xray.get('ai_results')
                                                if current_xray_ai_results and isinstance(current_xray_ai_results, list) and len(current_xray_ai_results) > 0:
                                                    class_counts = {}
                                                    for item in current_xray_ai_results:
                                                        if isinstance(item, dict) and 'class_name' in item:
                                                            class_name_english = item.get('class_name')
                                                            class_name_translated = Chinese_Name_Mapping.get(class_name_english, class_name_english)
                                                            class_counts[class_name_translated] = class_counts.get(class_name_translated, 0) + 1
                                                    if class_counts:
                                                        for name, count in class_counts.items():
                                                            st.write(f"- {name}: {count} 个")
                                                    else:
                                                        st.info("本次识别未检测到任何病变。")
                                                else:
                                                    st.info("没有识别结果可供统计。")
                                                
                                            else:
                                                st.info("该X光片未进行AI识别或识别失败。")
                                        except Exception as e:
                                            st.error(f"无法加载或显示图片 {xray['filename']}: {e}")
                                    else:
                                        st.warning(f"X光片文件 {xray['filename']} 未找到在服务器。")
                            
                        if st.session_state['xray_to_delete_confirm_id'] is not None:
                            xray_id_to_process = st.session_state['xray_to_delete_confirm_id']
                            xray_to_delete_obj = next((xr for xr in sorted_xrays if xr['id'] == xray_id_to_process), None)

                            if xray_to_delete_obj:
                                st.warning(f"确定要删除X光片 '{xray_to_delete_obj['filename']}' 吗？此操作不可撤销！")
                                col_confirm, col_cancel = st.columns(2)
                                with col_confirm:
                                    if st.button("✅ 确认删除", key='confirm_delete_final_btn'):
                                        patients_data_for_update = load_patients_data()
                                        for i, p in enumerate(patients_data_for_update['patients']):
                                            if p['id'] == selected_patient['id']:
                                                patients_data_for_update['patients'][i]['xrays'] = [
                                                    xr for xr in p['xrays'] if xr['id'] != xray_id_to_process
                                                ]
                                                break
                                        save_patients_data(patients_data_for_update)

                                        file_to_delete_path = os.path.join(UPLOAD_FOLDER, xray_to_delete_obj['filename'])
                                        if os.path.exists(file_to_delete_path):
                                            try:
                                                os.remove(file_to_delete_path)
                                                st.success(f"文件 {xray_to_delete_obj['filename']} 已从服务器删除。")
                                            except Exception as e:
                                                st.error(f"删除文件 {xray_to_delete_obj['filename']} 失败: {e}")
                                        else:
                                            st.warning(f"文件 {xray_to_delete_obj['filename']} 不存在于服务器，但已从记录中移除。")
                                        
                                        st.success(f"X光片记录 (ID: {xray_id_to_process}) 已成功删除。")
                                        st.session_state['xray_to_delete_confirm_id'] = None
                                        st.experimental_rerun()

                                with col_cancel:
                                    if st.button("❌ 取消删除", key='cancel_delete_final_btn'):
                                        st.session_state['xray_to_delete_confirm_id'] = None
                                        st.info("删除操作已取消。")
                                        st.experimental_rerun()
                            else:
                                st.session_state['xray_to_delete_confirm_id'] = None
                                st.warning("要删除的X光片未找到，可能已被删除。")
                                st.experimental_rerun()
                        


    elif selected_tab == "牙片图像识别系统":
        st.header("龋齿及牙周病变图像分割系统")
        st.write("上传一张牙齿X光图片，模型将尝试识别出龋齿等病变区域。此功能不关联患者数据。")

        uploaded_file = st.file_uploader("拖拽或点击上传文件", type=['png', 'jpg', 'jpeg', 'bmp'], key="ai_test_uploader")

        if uploaded_file is not None:
            temp_file_path = os.path.join(UPLOAD_FOLDER, "temp_test_upload_" + uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.image(uploaded_file) # REMOVED use_container_width

            if st.button("开始识别", key="start_ai_test_button"):
                with st.spinner('正在进行AI识别...'):
                    results_from_test = run_yolo_inference(temp_file_path)
                    
                    if results_from_test:
                        st.subheader("识别结果:")
                        original_image = Image.open(temp_file_path)
                        display_image_with_overlays(original_image, results_from_test)
                        
                        st.subheader("识别统计:")
                        if results_from_test and isinstance(results_from_test, list) and len(results_from_test) > 0:
                            class_counts = {}
                            for item in results_from_test:
                                if isinstance(item, dict) and 'class_name' in item:
                                    class_name_english = item.get('class_name')
                                    class_name_translated = Chinese_Name_Mapping.get(class_name_english, class_name_english)
                                    class_counts[class_name_translated] = class_counts.get(class_name_translated, 0) + 1

                            if class_counts:
                                st.write("本次识别共检测到以下病变：")
                                for name, count in class_counts.items():
                                    st.write(f"- {name}: {count} 个")
                            else:
                                st.info("本次识别未检测到任何病变。")
                        else:
                            st.info("没有识别结果可供统计。")
                        
                        st.json(results_from_test) # Optional: show raw JSON
                    else:
                        st.warning("AI识别未返回结果或发生错误。")
            
            os.remove(temp_file_path)
    

def admin_dashboard_page():
    """
    管理员仪表盘页面，包含用户管理和病历管理。
    """
    st.sidebar.image(resource_path('ini_image.png')) # REMOVED use_container_width
    st.sidebar.title(f"管理员 {st.session_state['username']}")
    logout_button()

    st.title("管理员仪表盘 - 系统管理")

    # 管理选项
    selected_tab = st.sidebar.radio("管理选项", ["用户管理", "病历管理"])

    if selected_tab == "用户管理":
        st.header("用户账户管理")
        
        users_df = load_users()
        st.dataframe(users_df) # REMOVED use_container_width

        st.subheader("编辑/删除用户")
        # 确保不能编辑/删除当前登录的管理员自己
        user_options = [u for u in users_df['username'].tolist() if u != st.session_state['username']] 
        
        if not user_options:
            st.info("除了当前管理员账户外，没有其他用户可管理。")
        else:
            selected_user_to_manage = st.selectbox("选择用户进行管理", user_options, key="admin_user_select")
            
            if selected_user_to_manage:
                st.markdown(f"**管理用户:** `{selected_user_to_manage}`")
                
                # 获取当前用户的角色和密码哈希
                current_user_data = users_df[users_df['username'] == selected_user_to_manage].iloc[0]
                current_user_role = current_user_data['role']

                # 编辑用户角色/密码
                st.markdown("##### 编辑用户角色/密码")
                new_role_for_user = st.selectbox("新角色", ["patient", "doctor", "admin"], 
                                                 index=["patient", "doctor", "admin"].index(current_user_role), 
                                                 key=f"edit_role_{selected_user_to_manage}")
                new_password_for_user = st.text_input("设置新密码 (留空则不修改)", type="password", key=f"edit_password_{selected_user_to_manage}")

                if st.button(f"保存对 {selected_user_to_manage} 的修改", key=f"save_user_edit_button_{selected_user_to_manage}"):
                    if update_user_in_csv(selected_user_to_manage, new_password_for_user if new_password_for_user else None, new_role_for_user):
                        st.success(f"用户 {selected_user_to_manage} 信息已更新！")
                        st.experimental_rerun()
                    else:
                        st.error(f"更新用户 {selected_user_to_manage} 失败。")

                # 删除用户
                st.markdown("##### 删除用户")
                if st.button(f"删除用户 {selected_user_to_manage}", key=f"delete_user_button_{selected_user_to_manage}"):
                    # 使用 Session State 实现二次确认
                    if st.session_state.get(f'confirm_delete_user_{selected_user_to_manage}', False):
                        if delete_user_from_csv(selected_user_to_manage):
                            st.success(f"用户 {selected_user_to_manage} 已成功删除。")
                            # 清除确认状态并重新运行
                            st.session_state[f'confirm_delete_user_{selected_user_to_manage}'] = False
                            st.experimental_rerun()
                        else:
                            st.error(f"删除用户 {selected_user_to_manage} 失败。")
                    else:
                        st.warning(f"确定要删除用户 '{selected_user_to_manage}' 吗？此操作不可撤销！")
                        st.session_state[f'confirm_delete_user_{selected_user_to_manage}'] = True
                        st.experimental_rerun()


    elif selected_tab == "病历管理":
        st.header("所有患者病历管理")
        patients_data_all = load_patients_data()
        all_patients = patients_data_all['patients']

        if not all_patients:
            st.info("系统内暂无患者记录。")
            return

        # 搜索所有患者
        search_query_admin_patient = st.text_input("搜索患者 (按姓名或联系方式)", key="admin_patient_search_query")
        filtered_patients_admin = []
        if search_query_admin_patient:
            search_query_lower = search_query_admin_patient.lower()
            for p in all_patients:
                if search_query_lower in p['name'].lower() or \
                   search_query_lower in p.get('contact', '').lower():
                    filtered_patients_admin.append(p)
        else:
            filtered_patients_admin = all_patients

        if not filtered_patients_admin:
            st.info("未找到匹配的患者。")
            return

        patient_options_admin = {p['name']: p['id'] for p in filtered_patients_admin}
        selected_patient_name_admin = st.selectbox("选择患者进行管理", list(patient_options_admin.keys()), key="admin_select_patient")
        
        if selected_patient_name_admin:
            selected_patient_id_admin = patient_options_admin[selected_patient_name_admin]
            selected_patient_admin = get_patient_by_id(selected_patient_id_admin) # 获取完整的患者数据

            if selected_patient_admin:
                # 管理员可以编辑任何患者的病历 (复用 patient_edit_form)
                edit_mode_key_admin_patient = f'admin_edit_mode_patient_{selected_patient_admin["id"]}'
                if edit_mode_key_admin_patient not in st.session_state:
                    st.session_state[edit_mode_key_admin_patient] = False
                
                # 重置编辑模式如果患者选择发生变化
                # (这个逻辑在 Streamlit 1.35.0+ 版本的 st.selectbox 中可能不再严格需要，但作为安全措施可保留)
                if 'admin_current_selected_patient_id' not in st.session_state:
                    st.session_state['admin_current_selected_patient_id'] = None
                if st.session_state['admin_current_selected_patient_id'] != selected_patient_admin['id']:
                    st.session_state['admin_current_selected_patient_id'] = selected_patient_admin['id']
                    st.session_state[edit_mode_key_admin_patient] = False
                    # st.experimental_rerun() # 如果需要立即清除旧编辑表单，可以在这里调用

                if st.session_state[edit_mode_key_admin_patient]:
                    # 管理员可以编辑，传入原始医生用户名
                    patient_edit_form(selected_patient_admin, selected_patient_admin.get('doctor_username', '无关联医生'))
                else:
                    st.subheader(f"患者信息: {selected_patient_admin['name']}")
                    if st.button("编辑病历信息 (管理员)", key=f"admin_edit_patient_button_{selected_patient_admin['id']}"):
                        st.session_state[edit_mode_key_admin_patient] = True
                        st.experimental_rerun()

                    # 显示患者基本信息和病历详情
                    st.write(f"**姓名:** {selected_patient_admin['name']}")
                    st.write(f"**性别:** {selected_patient_admin['gender']}")
                    st.write(f"**出生日期:** {selected_patient_admin['dob']}")
                    st.write(f"**联系方式:** {selected_patient_admin['contact']}")
                    st.write(f"**关联医生:** {selected_patient_admin.get('doctor_username', '无关联医生')}")
                    
                    st.markdown("---")
                    st.subheader("病历详情")
                    st.write(f"**主诉:** {selected_patient_admin.get('chief_complaint', 'N/A')}")
                    st.write(f"**现病史:** {selected_patient_admin.get('present_illness', 'N/A')}")
                    st.write(f"**既往史:** {selected_patient_admin.get('past_history', 'N/A')}")
                    st.write(f"**检查信息:** {selected_patient_admin.get('examination_info', 'N/A')}")
                    st.write(f"**鉴别诊断:** {selected_patient_admin.get('differential_diagnosis', 'N/A')}")
                    st.write(f"**治疗计划:** {selected_patient_admin.get('treatment_plan', 'N/A')}")
                    st.markdown("---")

                    # 导出病历按钮 (复用医生仪表盘功能)
                    if st.button("导出病历 (CSV)", key=f"admin_export_patient_record_csv_button_{selected_patient_admin['id']}"):
                        csv_data = generate_patient_record_csv(selected_patient_admin)
                        if csv_data:
                            export_filename = f"{selected_patient_admin['name']}_病历_{datetime.date.today().strftime('%Y%m%d')}.csv"
                            st.download_button(
                                label="点击下载病历CSV",
                                data=csv_data,
                                file_name=export_filename,
                                mime="text/csv",
                                key=f"admin_download_csv_button_{selected_patient_admin['id']}" # Ensure unique key for admin context
                            )
                            st.success("病历CSV文件已生成，请点击下载按钮。")
                        else:
                            st.warning("未能生成病历CSV数据。")

                    # 删除患者档案 (管理员专属)
                    if st.button("删除患者档案 (管理员)", key=f"admin_delete_patient_button_{selected_patient_admin['id']}"):
                        if st.session_state.get(f'admin_confirm_delete_patient_{selected_patient_admin["id"]}', False):
                            # 执行删除操作
                            patients_data_to_delete = load_patients_data()
                            patients_data_to_delete['patients'] = [p for p in patients_data_to_delete['patients'] if p['id'] != selected_patient_admin['id']]
                            save_patients_data(patients_data_to_delete)

                            # 删除关联的 X 光片文件
                            if selected_patient_admin.get('xrays'):
                                for xray in selected_patient_admin['xrays']:
                                    xray_file_path = os.path.join(UPLOAD_FOLDER, xray['filename'])
                                    if os.path.exists(xray_file_path):
                                        try:
                                            os.remove(xray_file_path)
                                            st.success(f"已删除X光片文件: {xray['filename']}")
                                        except Exception as e:
                                            st.error(f"删除X光片文件 {xray['filename']} 失败: {e}")
                                    else:
                                        st.warning(f"X光片文件 {xray['filename']} 不存在于服务器，但已从记录中移除。")

                            st.success(f"患者 {selected_patient_admin['name']} 档案已成功删除。")
                            st.session_state[f'admin_confirm_delete_patient_{selected_patient_admin["id"]}'] = False
                            st.experimental_rerun()
                        else:
                            st.warning(f"确定要删除患者 '{selected_patient_admin['name']}' 的所有档案吗？此操作不可撤销！")
                            st.session_state[f'admin_confirm_delete_patient_{selected_patient_admin["id"]}'] = True
                            st.experimental_rerun()
                            
                    st.subheader("历史X光片")
                    # Admin view of X-rays: simple list, no individual delete/expand (to avoid complexity, admin can delete the whole patient)
                    if selected_patient_admin.get('xrays'):
                        for xray in selected_patient_admin['xrays']:
                            image_path_admin_xray = os.path.join(UPLOAD_FOLDER, xray['filename'])
                            if os.path.exists(image_path_admin_xray):
                                st.markdown(f"- **文件:** {xray['filename']} (上传日期: {xray['upload_date']})")
                                # Optionally display image directly without expander for admin list view
                                img_admin_xray = Image.open(image_path_admin_xray)
                                display_image_with_overlays(img_admin_xray.copy(), xray.get('ai_results', []), show_confidence=False)
                            else:
                                st.warning(f"- X光片文件 {xray['filename']} 未找到在服务器。")
                        st.markdown("---")
                    else:
                        st.info("该患者暂无X光片。")
            else:
                st.error("未找到选定的患者数据。")

def patient_dashboard_page():
    st.sidebar.image(resource_path('ini_image.png')) # REMOVED use_container_width
    st.sidebar.title(f"患者 {st.session_state['username']}")
    logout_button()

    st.title("用户端")

    # Patient navigation tabs
    selected_tab = st.sidebar.radio("导航", ["我的病历信息", "我的X光片", "X光识别程序", "AI智能医生"])

    patient_data = get_patient_by_username(st.session_state['username'])
    if not patient_data:
        st.warning("您的档案信息未找到。请联系医生为您创建档案。")
        return

    # --- Tab 1: 我的病历信息 ---
    if selected_tab == "我的病历信息":
        st.header("我的档案信息")
        st.write(f"**姓名:** {patient_data['name']}")
        st.write(f"**性别:** {patient_data['gender']}")
        st.write(f"**出生日期:** {patient_data['dob']}")
        st.write(f"**联系方式:** {patient_data['contact']}")
        st.write(f"**关联医生:** {patient_data.get('doctor_username', '无关联医生')}") # 显示关联医生
        
        st.markdown("---")
        st.subheader("我的病历详情")
        st.write(f"**主诉:** {patient_data.get('chief_complaint', 'N/A')}")
        st.write(f"**现病史:** {patient_data.get('present_illness', 'N/A')}")
        st.write(f"**既往史:** {patient_data.get('past_history', 'N/A')}")
        st.write(f"**检查信息:** {patient_data.get('examination_info', 'N/A')}")
        st.write(f"**鉴别诊断:** {patient_data.get('differential_diagnosis', 'N/A')}")
        st.write(f"**治疗计划:** {patient_data.get('treatment_plan', 'N/A')}")
        st.markdown("---")

    # --- Tab 2: 我的X光片 ---
    elif selected_tab == "我的X光片":
        st.header("我的X光片")

        if patient_data.get('xrays'):
            sorted_xrays = sorted(patient_data['xrays'], key=lambda x: x['upload_date'], reverse=True)
            for xray in sorted_xrays:
                # Removed key from expander (as per last instruction)
                with st.expander(f"X光片: {xray['filename']} (上传日期: {xray['upload_date']})", expanded=False):
                    image_path = os.path.join(UPLOAD_FOLDER, xray['filename'])
                    if os.path.exists(image_path):
                        try:
                            img_original = Image.open(image_path)

                            # Always display original image
                            st.markdown("##### 原始X光片")
                            st.image(img_original, caption=f"原始X光片: {xray['filename']}") # REMOVED use_container_width
                            
                            # For patients, DO NOT display AI recognized image with overlays in history
                            # Optionally, you could show a message here that AI results are available for doctors
                            # st.info("AI识别结果仅供医生查看。") 

                        except Exception as e:
                            st.error(f"无法加载或显示图片 {xray['filename']}: {e}")
                    else:
                        st.warning(f"X光片文件 {xray['filename']} 未找到在服务器。")
        else:
            st.info("您暂无X光片记录。")

    # --- Tab 3: X光识别程序 (NEW) ---
    elif selected_tab == "X光识别程序":
        st.header("X光识别程序")
        st.write("上传一张牙齿X光图片，模型将尝试识别出龋齿等病变区域。此功能不关联您的病历档案。")

        # 使用一个唯一的key来区分患者识别测试上传器
        uploaded_file_patient = st.file_uploader("拖拽或点击上传文件", type=['png', 'jpg', 'jpeg', 'bmp'], key="patient_ai_test_uploader")

        if uploaded_file_patient is not None:
            # 保存临时文件用于推理
            temp_file_path_patient = os.path.join(UPLOAD_FOLDER, "temp_patient_upload_" + uploaded_file_patient.name)
            with open(temp_file_path_patient, "wb") as f:
                f.write(uploaded_file_patient.getbuffer())

            st.image(uploaded_file_patient) # REMOVED use_container_width

            # 使用一个唯一的key来区分患者识别测试按钮
            if st.button("开始识别", key="patient_start_ai_test_button"):
                with st.spinner('正在进行AI识别...'):
                    results_from_patient_test = run_yolo_inference(temp_file_path_patient)
                    
                    if results_from_patient_test is None: # True error occurred during inference
                        st.error("AI识别过程中出现错误，请检查日志。")
                    elif not results_from_patient_test: # Inference ran, but no detections were found
                        st.info("本次识别未检测到任何病变。")
                        original_image_patient = Image.open(temp_file_path_patient)
                        st.image(original_image_patient) # REMOVED use_container_width
                    else: # Inference ran, and detections were found
                        st.subheader("识别结果:")
                        original_image_patient_test = Image.open(temp_file_path_patient)
                        display_image_with_overlays(original_image_patient_test, results_from_patient_test, show_confidence=True) # Show confidence for test

                        st.subheader("识别统计:")
                        if results_from_patient_test and isinstance(results_from_patient_test, list) and len(results_from_patient_test) > 0:
                            class_counts = {}
                            for item in results_from_patient_test:
                                if isinstance(item, dict) and 'class_name' in item:
                                    class_name_english = item.get('class_name')
                                    class_name_translated = Chinese_Name_Mapping.get(class_name_english, class_name_english)
                                    class_counts[class_name_translated] = class_counts.get(class_name_translated, 0) + 1

                            if class_counts:
                                st.write("本次识别共检测到以下病变：")
                                for name, count in class_counts.items():
                                    st.write(f"- {name}: {count} 个")
                            else:
                                st.info("本次识别未检测到任何病变。")
                        else:
                            st.info("没有识别结果可供统计。")
                        
                        st.json(results_from_patient_test) # Optional: show raw JSON
            
            os.remove(temp_file_path_patient) # Delete temporary file

    # --- Tab 4: AI智能医生 (修改点) ---
    elif selected_tab == "AI智能医生":
        # 调用 AI 对话函数, 并将当前患者的数据传入
        chat_interface_page("患者", patient_data=patient_data)


# --- 主应用逻辑 ---
# 确保 session_state 初始化在脚本每次运行的开始
print("\n--- Script Start: Initializing session_state ---") # 调试信息
st.session_state['logged_in'] = st.session_state.get('logged_in', False)
st.session_state['username'] = st.session_state.get('username', None)
st.session_state['role'] = st.session_state.get('role', None)
print(f"session_state after init: logged_in={st.session_state['logged_in']}, username={st.session_state['username']}, role={st.session_state['role']}") # 调试信息
print("--- Session_state initialization complete ---") # 调试信息


# 根据登录状态显示不同的页面
if st.session_state['logged_in']:
    print(f"User is logged in: {st.session_state['username']} as {st.session_state['role']}") # 调试信息
    if st.session_state['role'] == 'doctor':
        doctor_dashboard_page()
    elif st.session_state['role'] == 'patient':
        patient_dashboard_page()
    elif st.session_state['role'] == 'admin':
        admin_dashboard_page()
    else:
        st.error("未知用户角色，请重新登录。")
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['role'] = None
        st.experimental_rerun()
else:
    print("User is not logged in. Showing login page.") # 调试信息
    login_page()
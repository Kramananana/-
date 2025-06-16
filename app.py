import os
import datetime
import re
import calendar
from collections import defaultdict
from functools import wraps
from urllib.parse import quote

from flask import (Flask, render_template, request, redirect, url_for,
                   flash, session, jsonify, Response, send_from_directory, make_response)
from werkzeug.utils import secure_filename
from PIL import Image
from sqlalchemy import desc

import config
import helpers # 确保导入了 helpers
from models import db, User, Patient, XRay, Appointment, Prescription, bcrypt

app = Flask(__name__)
app.config.from_object('config')

db.init_app(app)
bcrypt.init_app(app)

with app.app_context():
    helpers.init_database(app)
    helpers.load_models()

# 将 helpers 模块注入到所有模板的上下文中
@app.context_processor
def inject_helpers():
    return dict(helpers=helpers)

# --- 装饰器 ---
def login_required(f):
    """确保用户已登录"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('请先登录以访问此页面。', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def role_required(required_role):
    """确保用户拥有特定角色"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if session.get('user', {}).get('role') != required_role:
                flash(f'您没有权限访问此页面，需要 "{required_role}" 角色。', 'danger')
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# --- 核心路由 ---
@app.route('/')
def index():
    """根据用户角色重定向到对应的仪表盘"""
    if 'user' not in session:
        return redirect(url_for('login'))
    
    role = session['user']['role']
    if role == 'admin': return redirect(url_for('admin_dashboard'))
    if role == 'doctor': return redirect(url_for('doctor_dashboard'))
    if role == 'patient': return redirect(url_for('patient_dashboard'))
    
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """处理登录和注册逻辑"""
    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form.get('username')
        password = request.form.get('password')

        if action == 'login':
            user = helpers.verify_user(username, password)
            if user:
                session['user'] = {'id': user.id, 'username': user.username, 'role': user.role}
                flash(f'欢迎回来, {user.username}!', 'success')
                return redirect(url_for('index'))
            else:
                flash('用户名或密码错误。', 'danger')

        elif action == 'register':
            confirm_password = request.form.get('confirm_password')
            role = request.form.get('role')
            if not all([username, password, confirm_password, role]):
                flash('所有带 * 的字段均为必填项。', 'warning')
            elif password != confirm_password:
                flash('两次输入的密码不一致。', 'danger')
            elif role == 'doctor' and request.form.get('registration_code', '') != app.config['DOCTOR_REGISTRATION_CODE']:
                flash('医生专用注册码不正确。', 'danger')
            else:
                success, message = helpers.add_user(username, password, role)
                flash(message, 'success' if success else 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.pop('user', None)
    flash('您已成功登出。', 'info')
    return redirect(url_for('login'))

# --- 管理员功能 ---
@app.route('/admin/dashboard')
@login_required
@role_required('admin')
def admin_dashboard():
    """管理员仪表盘：管理所有用户和病历"""
    search_user_query = request.args.get('search_user', '')
    search_patient_query = request.args.get('search_patient', '')
    
    users_query = User.query
    if search_user_query:
        users_query = users_query.filter(User.username.ilike(f'%{search_user_query}%'))
        
    patients_query = Patient.query
    if search_patient_query:
        patients_query = patients_query.filter(db.or_(
            Patient.name.ilike(f'%{search_patient_query}%'),
            Patient.contact.ilike(f'%{search_patient_query}%')
        ))
        
    return render_template('admin_dashboard.html', 
                           users=users_query.all(), 
                           patients=patients_query.all(),
                           search_user_query=search_user_query,
                           search_patient_query=search_patient_query)

@app.route('/admin/user/update', methods=['POST'])
@login_required
@role_required('admin')
def admin_update_user():
    username = request.form.get('username')
    new_password = request.form.get('new_password')
    new_role = request.form.get('new_role')
    
    if helpers.update_user(username, new_password if new_password else None, new_role):
        flash(f'用户 {username} 的信息已成功更新。', 'success')
    else:
        flash(f'更新用户 {username} 失败，用户可能不存在。', 'danger')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/user/delete', methods=['POST'])
@login_required
@role_required('admin')
def admin_delete_user():
    username = request.form.get('username')
    if username == session['user']['username']:
        flash('操作被禁止：不能删除自己的账户！', 'danger')
    elif helpers.delete_user(username):
        flash(f'用户 {username} 及其所有关联数据（病历、预约等）已被成功删除。', 'success')
    else:
        flash(f'删除用户 {username} 失败，用户可能不存在。', 'danger')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/patient/delete', methods=['POST'])
@login_required
@role_required('admin')
def admin_delete_patient():
    patient_id = request.form.get('patient_id')
    if helpers.delete_patient(patient_id):
        flash(f'患者档案 (ID: {patient_id}) 已被成功删除。', 'success')
    else:
        flash(f'删除患者档案 (ID: {patient_id}) 失败。', 'danger')
    return redirect(url_for('admin_dashboard'))

# --- 医生功能 ---
@app.route('/doctor/dashboard')
@login_required
@role_required('doctor')
def doctor_dashboard():
    """医生仪表盘：管理自己的患者"""
    search_query = request.args.get('search', '')
    doctor = helpers.get_user_by_username(session['user']['username'])
    
    my_patients_query = doctor.patients # 使用模型中定义的反向关系
    if search_query:
        my_patients_query = my_patients_query.filter(
            db.or_(
                Patient.name.ilike(f'%{search_query}%'),
                Patient.contact.ilike(f'%{search_query}%')
            )
        )
    return render_template('doctor_dashboard.html', patients=my_patients_query.all(), search_query=search_query)

@app.route('/doctor/patient/add', methods=['POST'])
@login_required
@role_required('doctor')
def add_new_patient():
    patient_data = request.form.to_dict()
    patient_data['doctor_username'] = session['user']['username']
    
    if not patient_data.get('name'):
        flash('患者姓名是必填项。', 'danger')
    else:
        patient, message = helpers.add_patient(patient_data)
        flash(message, 'success' if patient else 'danger')
    return redirect(url_for('doctor_dashboard'))

@app.route('/doctor/patient/<int:patient_id>', methods=['GET', 'POST'])
@login_required
@role_required('doctor')
def patient_detail(patient_id):
    """查看和编辑患者病历详情"""
    patient = helpers.get_patient_by_id(patient_id)
    if not patient or patient.doctor.username != session['user']['username']:
        flash('找不到该患者或您没有权限访问。', 'danger')
        return redirect(url_for('doctor_dashboard'))
    
    if request.method == 'POST':
        if helpers.update_patient(patient_id, request.form.to_dict()):
            flash('患者信息更新成功！', 'success')
            return redirect(url_for('patient_detail', patient_id=patient_id))
        else:
            flash('更新失败，请重试。', 'danger')
            
    # 【修复】在后端完成排序
    xrays = patient.xrays.order_by(desc(XRay.upload_date)).all()
    prescriptions = patient.prescriptions.order_by(desc(Prescription.date_issued)).all()

    return render_template('patient_detail.html', 
                           patient=patient,
                           xrays=xrays, # 传递排序后的结果
                           prescriptions=prescriptions, # 传递排序后的结果
                           Chinese_Name_Mapping=helpers.Chinese_Name_Mapping)

@app.route('/doctor/patient/<int:patient_id>/xray/upload', methods=['POST'])
@login_required
@role_required('doctor')
def upload_xray(patient_id):
    """上传X光片，进行AI分析并保存"""
    if 'xray_file' not in request.files or not request.files['xray_file'].filename:
        flash('没有选择文件或文件无效。', 'warning')
        return redirect(url_for('patient_detail', patient_id=patient_id))
    
    file = request.files['xray_file']
    try:
        original_filename = secure_filename(file.filename)
        unique_filename = f"{patient_id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{original_filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # 运行AI分析
        ai_results = helpers.run_yolo_inference(file_path)
        
        # 绘制并保存带标注的图片
        img_original = Image.open(file_path)
        img_with_overlays = helpers.draw_overlays_on_image(img_original, ai_results)
        base, ext = os.path.splitext(unique_filename)
        overlay_filename = f"overlay_{base}.png"
        overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
        img_with_overlays.save(overlay_path)

        # 保存记录到数据库
        xray_data = {
            'filename': unique_filename,
            'upload_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ai_results': ai_results
        }
        if helpers.add_xray_to_patient(patient_id, xray_data):
            flash('X光片上传并识别成功！', 'success')
        else:
            flash('X光片信息保存失败。', 'danger')
    except Exception as e:
        # 在控制台打印详细错误，方便调试
        print(f"上传或处理X光片时发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        flash('处理文件时发生严重错误，请联系管理员。', 'danger')
        
    return redirect(url_for('patient_detail', patient_id=patient_id))

@app.route('/doctor/xray/<int:xray_id>/delete', methods=['POST'])
@login_required
@role_required('doctor')
def delete_xray(xray_id):
    xray = XRay.query.get_or_404(xray_id)
    # 权限检查
    if xray.patient.doctor.username != session['user']['username']:
        flash('您没有权限删除此X光片。', 'danger')
        return redirect(request.referrer or url_for('doctor_dashboard'))
        
    patient_id = xray.patient_id
    if helpers.delete_xray_from_patient(xray_id):
        flash('X光片已成功删除。', 'success')
    else:
        flash('删除X光片失败。', 'danger')
    return redirect(url_for('patient_detail', patient_id=patient_id))


@app.route('/doctor/appointments')
@login_required
@role_required('doctor')
def doctor_appointments():
    """医生查看预约日历"""
    doctor = helpers.get_user_by_username(session['user']['username'])
    try:
        year = int(request.args.get('year', datetime.datetime.now().year))
        month = int(request.args.get('month', datetime.datetime.now().month))
    except ValueError:
        year, month = datetime.datetime.now().year, datetime.datetime.now().month

    _, num_days = calendar.monthrange(year, month)
    start_date = datetime.date(year, month, 1)
    end_date = datetime.date(year, month, num_days)
    
    appointments_this_month = Appointment.query.filter(
        Appointment.doctor_id == doctor.id,
        db.func.date(Appointment.appointment_time) >= start_date,
        db.func.date(Appointment.appointment_time) <= end_date
    ).order_by(Appointment.appointment_time).all()


    appointments_by_day = defaultdict(list)
    for appt in appointments_this_month:
        appointments_by_day[appt.appointment_time.day].append(appt)

    cal = calendar.Calendar(firstweekday=6) # 周日为第一天
    month_calendar = cal.monthdatescalendar(year, month)
    
    prev_month_date = start_date - datetime.timedelta(days=1)
    next_month_date = end_date + datetime.timedelta(days=1)

    return render_template(
        'doctor_appointments.html', year=year, month=month,
        month_calendar=month_calendar, appointments_by_day=appointments_by_day,
        today=datetime.date.today(),
        prev_month_url=url_for('doctor_appointments', year=prev_month_date.year, month=prev_month_date.month),
        next_month_url=url_for('doctor_appointments', year=next_month_date.year, month=next_month_date.month)
    )

@app.route('/doctor/appointment/<int:appointment_id>')
@login_required
@role_required('doctor')
def appointment_detail_doctor_view(appointment_id):
    """医生查看预约详情"""
    appointment = helpers.get_appointment_by_id(appointment_id)
    if not appointment or appointment.doctor.username != session['user']['username']:
        flash('找不到该预约或您没有权限访问。', 'danger')
        return redirect(url_for('doctor_appointments'))
    return render_template('appointment_detail_doctor_view.html', appointment=appointment)

@app.route('/doctor/appointment/update_status', methods=['POST'])
@login_required
@role_required('doctor')
def update_appointment_status():
    """更新预约状态"""
    appointment_id = request.form.get('appointment_id')
    status = request.form.get('status')
    
    appt = helpers.get_appointment_by_id(appointment_id)
    if not appt or appt.doctor.username != session['user']['username']:
        flash('权限不足，操作失败。', 'danger')
        return redirect(url_for('doctor_appointments'))

    if helpers.update_appointment_status(appointment_id, status):
        flash(f'预约状态已成功更新为“{status}”', 'success')
    else:
        flash('更新预约状态失败，无效的状态值。', 'danger')
    
    return redirect(url_for('appointment_detail_doctor_view', appointment_id=appointment_id))

@app.route('/doctor/patient/<int:patient_id>/prescription/new', methods=['GET', 'POST'])
@login_required
@role_required('doctor')
def new_prescription(patient_id):
    """为患者开具新处方"""
    patient = helpers.get_patient_by_id(patient_id)
    if not patient or patient.doctor.username != session['user']['username']:
        flash('找不到该患者或您没有权限访问。', 'danger')
        return redirect(url_for('doctor_dashboard'))

    if request.method == 'POST':
        prescription, message = helpers.add_prescription(
            form_data=request.form, 
            patient_id=patient.id, 
            doctor_username=session['user']['username']
        )
        if prescription:
            flash(message, 'success')
            return redirect(url_for('patient_detail', patient_id=patient_id))
        else:
            flash(message, 'danger')
            
    return render_template('prescription_form.html', patient=patient, prescription=None)

@app.route('/doctor/prescription/<int:prescription_id>/edit', methods=['GET', 'POST'])
@login_required
@role_required('doctor')
def edit_prescription(prescription_id):
    prescription = helpers.get_prescription_by_id(prescription_id)
    if not prescription or prescription.doctor.username != session['user']['username']:
        flash('找不到该处方或您没有权限访问。', 'danger')
        return redirect(url_for('doctor_dashboard'))

    if request.method == 'POST':
        updated, message = helpers.update_prescription(prescription_id, request.form)
        flash(message, 'success' if updated else 'danger')
        if updated:
            return redirect(url_for('patient_detail', patient_id=prescription.patient_id))

    return render_template('prescription_form.html', patient=prescription.patient, prescription=prescription)

@app.route('/doctor/prescription/<int:prescription_id>/delete', methods=['POST'])
@login_required
@role_required('doctor')
def delete_prescription(prescription_id):
    prescription = helpers.get_prescription_by_id(prescription_id)
    if not prescription or prescription.doctor.username != session['user']['username']:
        flash('权限不足，操作失败。', 'danger')
        return redirect(request.referrer or url_for('doctor_dashboard'))
    
    patient_id = prescription.patient_id
    if helpers.delete_prescription(prescription_id):
        flash('处方已成功删除。', 'success')
    else:
        flash('删除处方失败。', 'danger')
    return redirect(url_for('patient_detail', patient_id=patient_id))

@app.route('/doctor/patient/<int:patient_id>/dental_chart')
@login_required
@role_required('doctor')
def dental_chart(patient_id):
    """牙位图标注界面"""
    patient = helpers.get_patient_by_id(patient_id)
    if not patient or patient.doctor.username != session['user']['username']:
        flash('找不到该患者或您没有权限访问。', 'danger')
        return redirect(url_for('doctor_dashboard'))
    
    return render_template('dental_chart.html', patient=patient, dental_chart_data=patient.dental_chart)

@app.route('/doctor/patient/<int:patient_id>/dental_chart/save', methods=['POST'])
@login_required
@role_required('doctor')
def save_dental_chart(patient_id):
    """保存牙位图数据"""
    patient = helpers.get_patient_by_id(patient_id)
    if not patient or patient.doctor.username != session['user']['username']:
        return jsonify({'status': 'error', 'message': '权限不足'}), 403
    
    patient.dental_chart = request.get_json()
    db.session.commit()
    return jsonify({'status': 'success', 'message': '牙位图已实时保存！'})

# --- 患者功能 ---
@app.route('/patient/dashboard')
@login_required
@role_required('patient')
def patient_dashboard():
    """患者仪表盘：查看个人病历、X光片和预约"""
    patient = helpers.get_patient_by_name(session['user']['username'])
    
    if not patient:
        first_doctor = User.query.filter_by(role='doctor').first()
        if first_doctor:
            patient, _ = helpers.add_patient({
                'name': session['user']['username'], 
                'doctor_username': first_doctor.username
            })
            flash(f'我们为您创建了新的个人档案，并关联到 {first_doctor.username} 医生。', 'info')
        else:
            flash('系统中暂无医生，无法为您创建档案，请联系管理员。', 'warning')
            return render_template('patient_dashboard.html', patient=None)

    # 【修复】在后端完成排序，再传递给模板
    xrays = patient.xrays.order_by(desc(XRay.upload_date)).all()
    prescriptions = patient.prescriptions.order_by(desc(Prescription.date_issued)).all()
    
    return render_template('patient_dashboard.html', 
                           patient=patient,
                           xrays=xrays,
                           prescriptions=prescriptions)


@app.route('/patient/appointments', methods=['GET', 'POST'])
@login_required
@role_required('patient')
def patient_appointments():
    """患者管理自己的预约"""
    patient = helpers.get_patient_by_name(session['user']['username'])
    if not patient:
        flash('找不到您的档案，请联系管理员。', 'warning')
        return redirect(url_for('patient_dashboard'))

    if request.method == 'POST':
        appointment_data = {
            'patient_id': patient.id,
            'appointment_time': request.form.get('appointment_time'),
            'reason': request.form.get('reason')
        }
        _, message = helpers.add_appointment(appointment_data)
        flash(message, 'success')
        return redirect(url_for('patient_appointments'))
    
    # 【修复】在后端完成排序
    appointments = patient.appointments.order_by(desc(Appointment.appointment_time)).all()
    return render_template('patient_appointments.html', patient=patient, appointments=appointments)

@app.route('/export/my_record')
@login_required
@role_required('patient')
def export_my_record_csv():
    """患者导出自己的病历"""
    patient = helpers.get_patient_by_name(session['user']['username'])
    if not patient:
        flash('找不到您的病历记录。', 'danger')
        return redirect(url_for('patient_dashboard'))
    return redirect(url_for('export_patient_csv', patient_id=patient.id))

# --- 公共功能 ---
@app.route('/ai_test', methods=['GET', 'POST'])
@login_required
def ai_test_page():
    """独立的AI识别测试页面"""
    if request.method == 'POST':
        if 'file' not in request.files or not request.files['file'].filename:
            flash('没有选择文件', 'warning')
            return render_template('ai_test_page.html')

        file = request.files['file']
        temp_filename = "temp_test_" + secure_filename(file.filename)
        original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(original_filepath)
        
        ai_results = helpers.run_yolo_inference(original_filepath)
        
        img_original = Image.open(original_filepath)
        img_with_overlays = helpers.draw_overlays_on_image(img_original, ai_results)
        
        base, _ = os.path.splitext(temp_filename)
        overlay_filename = 'overlay_' + base + '.png'
        overlay_filepath = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
        img_with_overlays.save(overlay_filepath)
        
        return render_template('ai_test_page.html', 
                               results=ai_results, 
                               original_image=temp_filename, 
                               overlay_image=overlay_filename,
                               Chinese_Name_Mapping=helpers.Chinese_Name_Mapping)
    return render_template('ai_test_page.html')

@app.route('/chat')
@login_required
def chat():
    """智能问答页面"""
    username = session['user']['username']
    role = session['user']['role']
    history = helpers.load_chat_history(username, role)
    
    # 患者自动加载病历上下文
    system_context = ""
    if role == 'patient':
        patient = helpers.get_patient_by_name(username)
        if patient:
            system_context = helpers.format_medical_record_for_ai(patient)

    return render_template('chat.html', history=history, system_context=system_context, user_role=role)

@app.route('/chat/stream', methods=['POST'])
@login_required
def chat_stream():
    """处理聊天消息并返回流式响应"""
    data = request.json
    messages = data.get('messages', [])
    user_role = session['user']['role']

    # 如果是医生，并提及了患者，则加载该患者病历作为上下文
    if user_role == 'doctor':
        last_user_message = messages[-1]['content']
        match = re.search(r'(?:@|根据)\s*([^\s,，的]+)', last_user_message)
        if match:
            patient_name = match.group(1).strip()
            patient = helpers.get_patient_by_name(patient_name)
            if patient and patient.doctor.username == session['user']['username']:
                patient_context = helpers.format_medical_record_for_ai(patient)
                # 插入上下文消息，并标记以便前端可以特殊处理（如不显示）
                context_message = {"role": "system", "content": patient_context, "is_context": True}
                messages.insert(-1, context_message)
            else:
                def error_stream():
                    yield f"错误：在您的名下未找到名为 “{patient_name}” 的患者，请检查姓名是否正确。"
                return Response(error_stream(), mimetype='text/event-stream')

    return Response(helpers.get_deepseek_response_stream(messages), mimetype='text/event-stream')

@app.route('/chat/save', methods=['POST'])
@login_required
def save_chat():
    """保存聊天记录"""
    helpers.save_chat_history(
        session['user']['username'],
        session['user']['role'],
        request.json.get('history', [])
    )
    return jsonify({"status": "success"})

@app.route('/prescription/<int:prescription_id>/print')
@login_required
def print_prescription(prescription_id):
    """打印处方"""
    prescription = helpers.get_prescription_by_id(prescription_id)
    if not prescription:
        flash('找不到该处方。', 'danger')
        return redirect(url_for('index'))
    
    # 权限检查
    current_user = helpers.get_user_by_username(session['user']['username'])
    is_admin = current_user.role == 'admin'
    is_doctor = prescription.doctor_id == current_user.id
    is_patient = (current_user.role == 'patient' and 
                  prescription.patient.name == current_user.username)
    
    if not (is_admin or is_doctor or is_patient):
        flash('您没有权限查看此内容。', 'danger')
        return redirect(url_for('index'))
        
    return render_template('prescription_print.html', prescription=prescription)

@app.route('/export/patient/<int:patient_id>')
@login_required
def export_patient_csv(patient_id):
    """导出指定患者的病历为CSV"""
    patient = helpers.get_patient_by_id(patient_id)
    if not patient:
        flash('找不到该患者。', 'danger')
        return redirect(url_for('index'))

    # 权限检查
    current_user = helpers.get_user_by_username(session['user']['username'])
    is_admin = current_user.role == 'admin'
    is_doctor = patient.doctor_id == current_user.id
    is_patient = (current_user.role == 'patient' and patient.name == current_user.username)
    
    if not (is_admin or is_doctor or is_patient):
        flash('您没有权限导出该患者的病历。', 'danger')
        return redirect(url_for('index'))
    
    csv_data = helpers.generate_patient_record_csv(patient)
    if csv_data:
        response = make_response(csv_data)
        # 处理中文文件名编码问题
        filename = f"{patient.name}_病历_{datetime.date.today().strftime('%Y%m%d')}.csv"
        response.headers['Content-Disposition'] = f"attachment; filename*=UTF-8''{quote(filename)}"
        response.headers['Content-Type'] = 'text/csv; charset=utf-8-sig'
        return response
    
    flash('生成病历CSV文件失败。', 'danger')
    return redirect(request.referrer or url_for('index'))

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    """提供上传文件的访问路径"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

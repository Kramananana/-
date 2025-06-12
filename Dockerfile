# 使用官方Python运行时作为基础镜像
FROM python:3.9-buster 

# 设置工作目录
WORKDIR /app

# 将requirements.txt复制到容器中
COPY requirements.txt .

# NEW: 确保 pip 和 setuptools 是最新的，以避免安装问题
RUN pip install --no-cache-dir --upgrade pip setuptools

RUN apt-get update && apt-get install -y libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir streamlit==1.35.0 \
    && echo "DEBUG: Streamlit 1.35.0 installed." \
    && python -c "import streamlit as st; print(f'DEBUG: Container Streamlit Version: {st.__version__}')"

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有项目文件到容器的/app目录
COPY . . 

# 暴露Streamlit应用的默认端口
EXPOSE 8501

# 在容器启动时运行Streamlit应用
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=true", "--server.enableXsrfProtection=false"]

# 针对Docker部署的一些Streamlit配置
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLE_CORS=true
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
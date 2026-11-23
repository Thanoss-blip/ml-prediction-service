FROM python:3.10-slim
WORKDIR /app
COPY app/requirements.txt .
RUN pip install --no-cache-dir \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt
COPY app/ .
COPY model/ /model/
EXPOSE 5000
CMD ["python", "app.py"]

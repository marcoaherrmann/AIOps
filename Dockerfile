FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# xgboost first without GPU/CUDA extras, then the rest
RUN pip install --no-cache-dir xgboost==3.2.0 --no-deps
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888 5000

# Starts MLflow UI (port 5000) and Jupyter (port 8888)
CMD ["bash", "-c", "mlflow ui --host 0.0.0.0 --port 5000 & jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token='' --ServerApp.password=''"]
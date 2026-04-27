FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# xgboost first without GPU/CUDA extras, then the rest
RUN pip install --no-cache-dir xgboost==3.2.0 --no-deps
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make startup script executable
RUN chmod +x start.sh

EXPOSE 8888 5000

# Starts both Jupyter (port 8888) and MLflow UI (port 5000)
CMD ["./start.sh"]
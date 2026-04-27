FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install xgboost without GPU dependencies
RUN pip install --no-cache-dir xgboost==3.2.0 --no-deps
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888 5000

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
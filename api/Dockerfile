# api/Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY ../requirements.txt .  
RUN pip install -r requirements.txt
COPY ../api .
COPY ../best_mnist_cnn.pth /app/best_mnist_cnn.pth
COPY ../model ./model
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

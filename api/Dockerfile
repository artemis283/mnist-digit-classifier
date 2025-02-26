# api/Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY ../requirements.txt .  
RUN pip install -r requirements.txt
COPY ../api .  # Copy API folder
COPY ../model ./model
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

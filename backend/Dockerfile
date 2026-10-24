# Package the API with the same Python minor version used for local testing.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:7860", "app:superkart_api"]

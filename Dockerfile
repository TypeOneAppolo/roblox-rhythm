FROM python:3.11-slim
RUN apt-get update && apt-get install -y libsndfile1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]

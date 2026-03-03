FROM python:3.11

# Install system audio libraries
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg

# Set up user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY --chown=user . .

# Hugging Face runs on port 7860 by default
CMD ["python", "app.py"]

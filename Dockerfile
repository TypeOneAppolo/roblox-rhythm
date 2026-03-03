# Use a lightweight Python image
FROM python:3.11-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y libsndfile1

# Set the working directory
WORKDIR /app

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Run the app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
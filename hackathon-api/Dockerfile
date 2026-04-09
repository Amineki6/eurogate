# Use an official lightweight Python image
FROM python:3.9-slim

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Use the PORT environment variable injected by Cloud Run
# 'exec' ensures uvicorn handles shutdown signals correctly
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
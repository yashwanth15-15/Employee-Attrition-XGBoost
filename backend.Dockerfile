# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies (required for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
# We do this before copying the app to leverage Docker layer caching
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire backend application and models
COPY . /app/

# Ensure the SQLite data directory exists
RUN mkdir -p /app/data

# Create a non-root user for security
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose the API port (informational; Render uses $PORT)
EXPOSE 8000

# Command to run the application using Uvicorn
# Uses shell form so $PORT is expanded at runtime (Render injects $PORT)
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}

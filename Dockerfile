FROM python:3.11-slim

WORKDIR /app

# Install system dependencies yang diperlukan
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy aplikasi
COPY . .

# Create necessary directories
RUN mkdir -p uploads results flask_session

# Expose port
EXPOSE 5000

# Environment variables
ENV FLASK_APP=auto_indexing.py
ENV FLASK_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/ || exit 1

# Run aplikasi dengan gunicorn
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 wsgi:app"]

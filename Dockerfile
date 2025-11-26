# Use Microsoft's official Playwright image with Python and browsers pre-installed
FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p /app/logs

# Expose port (Railway/Render will set PORT env var)
EXPOSE 8000

# Start the application
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}

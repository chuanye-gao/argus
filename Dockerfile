FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY argus/ ./argus/
COPY samples/ ./samples/

VOLUME ["/data"]
EXPOSE 8000

# Default: start the HTTP API server
CMD ["python", "-m", "argus", "--serve", "--host", "0.0.0.0", "--port", "8000"]

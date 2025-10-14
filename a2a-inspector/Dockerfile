# Custom Dockerfile for a2a-inspector (adds build tools for cffi)
FROM python:3.11-slim

WORKDIR /app

# Install build tools for Python native extensions
RUN apt-get update && apt-get install -y build-essential gcc libffi-dev && rm -rf /var/lib/apt/lists/*

# Copy source code from the cloned repo
COPY . .

# Install dependencies (adapt to a2a-inspector's requirements)
RUN pip install --upgrade pip
RUN pip install -r requirements.txt || true
RUN pip install -r backend/requirements.txt || true

EXPOSE 8080

CMD ["python", "backend/main.py"]

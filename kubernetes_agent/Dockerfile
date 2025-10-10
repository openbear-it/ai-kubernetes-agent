FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Start the server with uvicorn for production
CMD ["uvicorn", "main:server", "--host", "0.0.0.0", "--port", "8080"]

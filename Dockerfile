FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects $PORT at runtime; default to 8080 for local `docker run`.
ENV PORT=8080
EXPOSE 8080

# Shell form so $PORT is expanded at container start (Cloud Run requirement).
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT}

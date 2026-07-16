FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app
COPY online /app/online
RUN pip install --no-cache-dir /app/online

EXPOSE 8000
CMD ["uvicorn", "online.api.app:app", "--host", "0.0.0.0", "--port", "8000"]


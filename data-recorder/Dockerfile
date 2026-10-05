# Build from the repo root:  docker build -f data-recorder/Dockerfile -t kraken-recorder .
FROM python:3.12-slim
WORKDIR /app
COPY data-recorder/requirements.txt data-recorder/requirements.txt
RUN pip install --no-cache-dir -r data-recorder/requirements.txt
COPY config config
COPY data-recorder data-recorder
WORKDIR /app/data-recorder
ENV RECORDER_OUT=/data PYTHONUNBUFFERED=1
VOLUME /data
CMD ["python", "-m", "recorder.main"]

# Hourly paper trader for the droplet. Build from the repo root:
#   docker build -f deploy/paper-trader/Dockerfile -t kraken-paper .
# scikit-learn is pinned to the version the registered models were pickled
# with (their meta records sklearn_version); bump it together with a retrain.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir psycopg2-binary requests joblib scikit-learn==1.9.1
COPY config config
COPY python-strategy python-strategy
WORKDIR /app/python-strategy
ENV PYTHONUNBUFFERED=1
CMD ["python", "scripts/paper_trade.py", "--arm", "all"]

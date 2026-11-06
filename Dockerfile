FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
ENV IWL_REDIS_URL=redis://redis:6379/0
ENV IWL_INFERENCE_MODE=MOCK

CMD ["inference-api"]

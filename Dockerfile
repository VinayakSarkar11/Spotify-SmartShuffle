FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/users logs

# Run from web/ so `from auth import ...` resolves correctly.
# __file__-based ROOT paths still resolve to /app since they use abspath.
WORKDIR /app/web

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --workers 1

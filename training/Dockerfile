FROM python:3.10-slim

WORKDIR /app

COPY train.py /app/
COPY requirements.txt /app/

RUN apt-get update && apt-get install -y \
    gcc \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "train.py"]
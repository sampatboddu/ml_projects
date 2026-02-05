FROM python:3.11-slim

WORKDIR /app

COPY /app/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY /app/main.py main.py

COPY /artifacts/model /model

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

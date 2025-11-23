FROM python:3.12-slim
WORKDIR /app

RUN pip install uv
RUN uv pip install -r requirements.txt --target /app/.venv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
FROM python:3.12-slim
COPY main.py .
COPY src/ src/

CMD ["python", "main.py"]
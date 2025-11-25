FROM python:3.12-slim

WORKDIR /app

# Install uv once, globally
RUN pip install --no-cache-dir uv

# Copy reuirements.txt for caching
COPY requirements.txt .

# Install deps into the system Python inside the container
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy  the source  code and main.py
COPY src/ src/
COPY main.py .

# Create directory for model persistence
RUN mkdir -p /app/models

# Set the volume mount point
VOLUME ["/app/models"]

CMD ["python", "main.py"]

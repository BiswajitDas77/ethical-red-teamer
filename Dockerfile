# Ethical Red-Teamer — OpenEnv Environment
# Dockerfile for Hugging Face Spaces deployment

FROM python:3.11-slim

# Metadata
LABEL maintainer="Biswajit"
LABEL description="Ethical Red-Teamer: AI Safety & Ethics OpenEnv Environment"

# Create non-root user (HF Spaces best practice)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source files
COPY models.py   /app/models.py
COPY tasks.py    /app/tasks.py
COPY server/     /app/server/
# client.py is optional and was omitted from upload
COPY openenv.yaml /app/openenv.yaml

# Create server __init__.py if missing
RUN touch /app/server/__init__.py

# Switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# HF Spaces expects port 7860
EXPOSE 7860

# Environment variables (override via HF Spaces secrets)
ENV PORT=7860
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

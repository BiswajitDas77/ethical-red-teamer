# Ethical Red-Teamer — OpenEnv Environment
# Dockerfile for Hugging Face Spaces deployment

FROM python:3.11-slim

# Metadata
LABEL maintainer="Biswajit"
LABEL description="Ethical Red-Teamer: AI Safety & Ethics OpenEnv Environment"

# Create non-root user (HF Spaces best practice)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy source files first (better layer caching)
COPY models.py tasks.py /app/
COPY server/ /app/server/
COPY openenv.yaml /app/openenv.yaml

# Install dependencies
RUN pip install --no-cache-dir -r /app/server/requirements.txt

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

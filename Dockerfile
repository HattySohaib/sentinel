# Dockerfile for Render Deployment

# ---- Stage 1: Build ----
# Use a slim Python image as a base for building our dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build essentials (optional but good practice)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: Final Image ----
# Use the same slim Python image for the final, lightweight container
FROM python:3.10-slim

# Create a non-root user for security
RUN groupadd --system app && useradd --system --gid app app
USER app

WORKDIR /home/app

# Copy the installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code and model files, and set ownership in one step
COPY --chown=app:app . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
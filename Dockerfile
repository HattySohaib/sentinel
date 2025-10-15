# Dockerfile for Render Deployment

# ---- Stage 1: Build ----
# Use a slim Python image as a base for building our dependencies
FROM python:3.10-slim AS builder

# Set the working directory
WORKDIR /app

# Install build essentials (sometimes needed for certain libraries)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Copy only the requirements file to leverage Docker's layer caching
COPY requirements.txt .

# Install all dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: Final Image ----
# Use the same slim Python image for the final, lightweight container
FROM python:3.10-slim

# Create a non-root user and group for security
RUN groupadd --system app && useradd --system --gid app app

# Set the working directory for the final application
WORKDIR /home/app

# Copy the installed packages and binaries from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Change ownership of the app directory to the app user
COPY --chown=app:app . .

# Switch to non-root user
USER app

# Copy the application code (main.py) and the model files
COPY . .

# Expose the port the app runs on. Render will detect this.
EXPOSE 8000

# Command to run the application using uvicorn
# The host 0.0.0.0 is crucial for it to be accessible from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# Use an official Python slim runtime as the base image
FROM python:3.10-slim

# Install system dependencies: Tesseract OCR and clean up apt caches
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with UID 1000 to comply with Hugging Face guidelines
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory in the user's home directory
WORKDIR $HOME/app

# Copy requirements and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application files
# We use a wildcard pattern Luma.mp[4] so the COPY command does not fail
# if the large video file was omitted from the git repository.
COPY --chown=user app.py .
COPY --chown=user Luma.mp[4] ./

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Start the application using a shell to resolve the PORT environment variable if present
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]

# Use an official Python 3.10 image as the base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app
# Create a /data directory
RUN mkdir -p /data
# Install Node.js and npm (using the official Node.js package)
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    curl -sL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean

# Copy the requirements.txt and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the front-end application files (optional, if you have them)
# COPY frontend /app/frontend

# Install Node.js dependencies for the front-end (optional, if you have them)
# RUN cd frontend && npm install

# Copy the rest of the application code
COPY . /app

# Expose the port that uvicorn will run on
EXPOSE 8000

# Command to run the FastAPI app with uvicorn
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

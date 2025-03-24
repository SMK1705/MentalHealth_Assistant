# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variable to force unbuffered stdout and stderr (helpful for logging)
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose any necessary ports (if your app requires them)
EXPOSE 8000

# Define the default command to run your application.
CMD ["python", "main.py"]
# Use an official Python runtime as a parent image.
# Pinned to 3.11 to match runtime.txt and the devcontainer, and to ensure
# prebuilt torch 2.6.0 wheels are available.
FROM python:3.11-slim

# Force unbuffered stdout and stderr (helpful for logging)
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Streamlit serves on 8501 by default.
EXPOSE 8501

# Run the Streamlit app (the shipped entry point). Shell form so ${PORT} is
# expanded for PaaS platforms that inject it; binds to 0.0.0.0 for the container.
CMD streamlit run app_chat.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.enableCORS false

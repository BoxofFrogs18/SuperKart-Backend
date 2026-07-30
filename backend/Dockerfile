
# Use a lightweight Python 3.9 image as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all backend files into the container
COPY . .

# Install the Python dependencies listed in requirements.txt
# without storing the pip cache, which helps reduce image size
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Start the Flask API using Gunicorn
# -w 4: Runs four worker processes
# -b 0.0.0.0:7860: Makes the API available on port 7860
# app:app: Loads the Flask object named "app" from app.py
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:app"]

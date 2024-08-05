# Use the official Python 3.12 image as the base
FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Copy the application code to the working directory
COPY . .

# Install the dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK tokenizer
RUN python -m nltk.downloader punkt

# Expose the port the app runs on
EXPOSE 8000

# Set the working directory inside src folder
WORKDIR /app/src

# Define the command to run the app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
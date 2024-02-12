# Use the official Tensorflow Docker image as base
FROM tensorflow/tensorflow:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the Python script containing your model code to the container
COPY flask_script_checkpoint.py /app/flask_script_checkpoint.py
COPY train.csv /app/train.csv
# Copying model checkpoints
COPY model_checkpoint.ckpt.data-00000-of-00001 /app/model_checkpoint.ckpt.data-00000-of-00001
COPY model_checkpoint.ckpt.index /app/model_checkpoint.ckpt.index
COPY requirements.txt .

#Installing dependent packages
RUN pip install -r requirements.txt --ignore-installed

# Expose the port your application will run on
EXPOSE 8178

# Set the command to run your script when the container starts
CMD ["python", "/app/flask_script_checkpoint.py"]
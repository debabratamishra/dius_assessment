# Using the official Tensorflow Docker image as base
FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY flask_serving.py /app/flask_serving.py
COPY train.csv /app/train.csv

# Copying model checkpoints
COPY model_checkpoint.ckpt.data-00000-of-00001 /app/model_checkpoint.ckpt.data-00000-of-00001
COPY model_checkpoint.ckpt.index /app/model_checkpoint.ckpt.index
COPY requirements.txt .

#Installing dependent packages
RUN pip install -r requirements.txt --ignore-installed

# Exposing the port your application will run on
EXPOSE 8178

# Setting the command to run the script when the container starts
CMD ["python", "/app/flask_serving.py"]
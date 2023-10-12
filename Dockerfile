FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git\
        python3-dev \
        python3-pip \
        build-essential

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Upgrade pip
RUN python3 -m pip install --upgrade pip


COPY ./mains ./mains
COPY ./models ./models
COPY ./schema ./schema
COPY ./app.py . 

EXPOSE 5000 

CMD uvicorn --host 0.0.0.0 --port 5000 app:app
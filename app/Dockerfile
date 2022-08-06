# Set base image (host OS)
FROM python:3.8

# By default, listen on port 80
EXPOSE 80/tcp

WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/


# Specify the command to run on container start
CMD [ "python", "./main.py" ]

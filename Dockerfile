FROM python:3.10-slim-buster

RUN apt-get update
RUN apt-get install nano

# Install GDAL dependencies
RUN apt-get update &&\
    apt-get install -y binutils libproj-dev gdal-bin

RUN mkdir wd
WORKDIR wd
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY app/ ./

CMD [ "gunicorn", "--workers=5", "--threads=1", "-b 0.0.0.0:80", "app:server"]
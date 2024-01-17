FROM python:3.10-slim-buster

RUN apt-get update
RUN apt-get install nano

# Update pip 
RUN pip install --upgrade pip

# Install GDAL dependencies
RUN sudo apt-get install libgdal-dev libgdal1h

# Export to C compilers
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal



RUN mkdir wd
WORKDIR wd
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY app/ ./

CMD [ "gunicorn", "--workers=5", "--threads=1", "-b 0.0.0.0:80", "app:server"]
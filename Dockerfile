FROM ghcr.io/osgeo/gdal:ubuntu-small-latest

RUN apt-get update
RUN apt-get install nano \
    python3-pip


RUN mkdir wd
WORKDIR wd
COPY requirements.txt .
RUN pip3 install --yes -r requirements.txt

COPY app/ ./

CMD [ "gunicorn", "--workers=5", "--threads=1", "-b 0.0.0.0:80", "app:server"]
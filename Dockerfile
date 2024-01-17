FROM geodata/gdal

RUN apt-get update
RUN apt-get install nano

RUN apt-get update && apt-get install -y python3-pip

RUN mkdir wd
WORKDIR wd
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY app/ ./

CMD [ "gunicorn", "--workers=5", "--threads=1", "-b 0.0.0.0:80", "app:server"]
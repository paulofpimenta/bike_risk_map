FROM python:3.10-slim-buster

RUN apt-get update


RUN mkdir wd
WORKDIR wd
COPY application/requirements.txt .
RUN pip3 install -r requirements.txt

COPY application/ ./

#CMD [ "gunicorn", "--workers=5", "--threads=1", "-b 0.0.0.0:80", "app:server"]
#CMD ["waitress-serve" "--host=0.0.0.0" "--port=80"  "appname:app.server"]
#CMD ["waitress-serve" "--host=0.0.0.0" "--port=80"  "appname:app.server"]
CMD [ "python3", "./app.py"]
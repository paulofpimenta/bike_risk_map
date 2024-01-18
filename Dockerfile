FROM python:3.10-slim-buster

#RUN add-apt-repository ppa:certbot/certbot
RUN apt-get update
RUN apt-get install -y --no-install-recommends python3-certbot-nginx
RUN pip3 install --upgrade pip

RUN letsencrypt certonly -a webroot --webroot-path=/letsencrypt -d app2.ouicodedata.com -d www.app2.ouicodedata.com

RUN mkdir wd

COPY application/requirements.txt .

COPY application/server-conf/nginx.conf /etc/nginx/

RUN pip3 install -r requirements.txt

COPY application/ ./

WORKDIR wd

#CMD [ "gunicorn", "--workers=5", "--threads=1", "-b 0.0.0.0:80", "app:server"]
#CMD ["waitress-serve" "--host=0.0.0.0" "--port=80"  "appname:app.server"]
#CMD ["waitress-serve" "--host=0.0.0.0" "appname:app.server", "port:5000", "url_scheme:https"]
#CMD [ "python3", "./app.py"]
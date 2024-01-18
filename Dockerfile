FROM python:3.10-slim-buster

#RUN add-apt-repository ppa:certbot/certbot
RUN apt-get update
RUN apt-get install -y python3-certbot-nginx supervisor
RUN pip3 install uwsgi


RUN letsencrypt certonly -a webroot --webroot-path=/var/www/app2.ouicodedata.com/html/ -d app2.ouicodedata.com -d www.app2.ouicodedata.com

RUN mkdir wd

COPY application/requirements.txt .

COPY application/server-conf/nginx.conf /etc/nginx/
COPY application/server-conf/uwsgi.ini /etc/uwsgi/
COPY application/server-conf/supervisord.conf /etc/supervisor/

RUN pip3 install -r requirements.txt

COPY application/ ./

WORKDIR wd

CMD ["/usr/bin/supervisord"]

#CMD [ "gunicorn", "--workers=5", "--threads=1", "-b 0.0.0.0:80", "app:server"]
#CMD ["waitress-serve" "--host=0.0.0.0" "--port=80"  "appname:app.server"]
#CMD ["waitress-serve" "--host=0.0.0.0" "appname:app.server", "port:5000", "url_scheme:https"]
#CMD [ "python3", "./app.py"]
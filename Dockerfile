FROM nginx:1.15-alpine

#RUN add-apt-repository ppa:certbot/certbot
RUN pip3 install --upgrade pip
#RUN pip3 install waitress

#RUN letsencrypt certonly -a webroot --webroot-path=/letsencrypt -d app2.ouicodedata.com -d www.app2.ouicodedata.com

RUN mkdir -p wd
WORKDIR wd

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY application ./

RUN rm /etc/nginx/conf.d/default.conf
COPY application/server-conf/nginx.conf /etc/nginx/

COPY ./certs/ /etc/letsencrypt


RUN pip3 install -r requirements.txt


RUN ls --recursive /wd/


#CMD [ "gunicorn", "--workers=5", "--threads=1", "-b 0.0.0.0:80", "app:server"]
#CMD ["waitress-serve" "--host=0.0.0.0" "--port=80"  "appname:app.server"]
#CMD ["waitress-serve","--host=0.0.0.0","--call","app:create_app", "port:5000", "url_scheme:https"]
CMD ["nginx", "-g", "daemon off;"]
CMD ["python3","app.py"]

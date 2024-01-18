FROM nginx/unit:1.23.0-python3.9


#RUN letsencrypt certonly -a webroot --webroot-path=/letsencrypt -d app2.ouicodedata.com -d www.app2.ouicodedata.com

RUN mkdir -p wd
WORKDIR wd

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY application ./

RUN rm /etc/nginx/conf.d/default.conf
COPY application/server-conf/nginx.conf /etc/nginx/

COPY ./certs/ /etc/letsencrypt


RUN apt update && apt install -y python3-pip                                 \
    && pip3 install -r requirements.txt                               
    && apt remove -y python3-pip                                              
    && apt autoremove --purge -y                                              
    && rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/*.list

RUN ls --recursive /wd/

EXPOSE 80

CMD ["python3","app.py"]

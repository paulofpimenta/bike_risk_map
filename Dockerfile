FROM tiangolo/uwsgi-nginx-flask:python3.10

COPY ./app /app
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# FROM tiangolo/uwsgi-nginx:python3.11

# WORKDIR /code

# COPY ./application/requirements.txt /code/requirements.txt

# RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# COPY ./application /code/application

# CMD ["gunicorn", "--conf", "app/server-conf/gunicorn_conf.py", "--bind", "0.0.0.0:80", "application.main:app"]


# Your Dockerfile code...

# FROM nginx/unit:1.23.0-python3.9


# #RUN letsencrypt certonly -a webroot --webroot-path=/letsencrypt -d app2.ouicodedata.com -d www.app2.ouicodedata.com

# RUN mkdir -p wd
# WORKDIR wd

# COPY application ./

# COPY application/server-conf/nginx.conf /etc/nginx/

# COPY ./certs/ /etc/letsencrypt


# RUN apt update && apt install -y python3-pip \
#     && pip3 install -r requirements.txt \                               
#     && apt remove -y python3-pip \                                              
#     && apt autoremove --purge -y \                                             
#     && rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/*.list

# RUN ls --recursive /wd/

# CMD ["nginx", "-g", "daemon off;"]

# EXPOSE 80

# CMD ["python3","app.py"]

FROM python:3.10

RUN mkdir -p /dash_app
WORKDIR /dash_app
COPY ./app/requirements.txt /dash_app
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app/ /dash_app/
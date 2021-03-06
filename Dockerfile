FROM tiangolo/uwsgi-nginx-flask:python3.7

COPY ./app /app
COPY requirements.txt /app

WORKDIR /app
RUN pip install -r requirements.txt
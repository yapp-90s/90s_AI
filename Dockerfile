FROM python:3.9

RUN pip3 install django

WORKDIR /usr/src/app

COPY . .

WORKDIR ./gugongs

CMD ["python3", "manage.py", "runserver", "0:8888"]

EXPOSE 8888

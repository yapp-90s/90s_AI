FROM python:3.9

COPY requirements.txt ./
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r requirements.txt

WORKDIR /usr/src/app

COPY . .

EXPOSE 8888


CMD ["python3", "manage.py", "runserver", "0.0.0.0:8888"]


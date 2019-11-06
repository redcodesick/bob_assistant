FROM python:3.7
COPY . /app
WORKDIR /app

ENV GOOGLE_APPLICATION_CREDENTIALS
ENV DATABASE_URL

RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y python-opencv

ENTRYPOINT [ "gunicorn", "-b", ":8080", "main:app", "--timeout", "600" ]

EXPOSE 8080
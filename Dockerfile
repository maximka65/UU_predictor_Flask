FROM python:3.6

RUN mkdir -p /app
WORKDIR /app/UU_predictor

COPY . /app

RUN pip3 install -r requirements.txt

#RUN python3 model_train.py

EXPOSE 5000

CMD ["gunicorn", "flask_server:app", "--bind", "0.0.0.0:5000", "-k", "gevent", "--worker-connections", "2"]

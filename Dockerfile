# Dockerfile - this is a comment. Delete me if you want.
FROM python:3.7
COPY . /app
ADD PossWordsDF.csv /var/
ADD LSTM_model_EscalationDetection.sav /var/
ADD Escalation_tok.sav /var/

ADD LSTM_model_AppreciationDetection.sav /var/
ADD tok.sav /var/

ADD requirements.txt /app/

WORKDIR /app

RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
CMD ["app.py"]

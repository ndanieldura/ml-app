FROM python:3.9

RUN pip install --upgrade pip 

RUN mkdir /app

WORKDIR /app  

ADD . . 

RUN pip install -r requirements.txt 

CMD streamlit run --server.port $PORT app.py
FROM tensorflow/tensorflow:1.7.0-py3
RUN pip install flask tensorflow-serving-api
RUN mkdir /app 
RUN chmod +x /app
WORKDIR /app
ADD . /app
EXPOSE 80
CMD python blerssi-webapp-client.py

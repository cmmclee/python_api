FROM python:3.6-alpine
RUN git clone https://github.com/aggresss/docker-opencv-python.git /code
WORKDIR /code
RUN pip install flask
EXPOSE 5000
CMD ["python", "app.py"]
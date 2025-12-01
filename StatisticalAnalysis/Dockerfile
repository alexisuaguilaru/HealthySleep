FROM python:3.13-slim
EXPOSE 8080 8181

WORKDIR /Analysis

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN useradd -m app_user
USER app_user
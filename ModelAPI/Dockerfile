FROM python:3.13-slim

EXPOSE 8000
ENV API_USER=api_user
ENV API_HOME=/home/$API_USER

RUN groupadd $API_USER && useradd -m -s /bin/bash -g $API_USER $API_USER

WORKDIR $API_HOME/ModelAPI
COPY --chown=$API_USER:$API_USER . .

RUN pip install --cache-dir=false -r requirements.txt

USER $API_USER

CMD ["fastapi", "run", "main.py"]
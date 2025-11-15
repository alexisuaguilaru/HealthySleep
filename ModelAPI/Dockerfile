FROM python:3.13-slim

EXPOSE 8000

WORKDIR /ModelAPI
COPY . .

RUN pip install --cache-dir=false -r requirements.txt

CMD ["fastapi", "run", "main.py"]
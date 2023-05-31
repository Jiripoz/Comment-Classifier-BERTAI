FROM python:3.9

COPY . /app

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

RUN python download_model.py

CMD ["python", "start.py"]

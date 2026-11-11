FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
COPY vendor/ ./vendor/
RUN pip install --no-cache-dir --no-index --find-links vendor -r requirements.txt
COPY *.py ./
COPY reader ./reader
COPY example_papers/ ./example_papers/
COPY tests/ ./tests/
ENTRYPOINT ["python", "paper.py"]

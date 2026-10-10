FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rag ./rag
COPY data/handbook ./data/handbook
# Models are downloaded on first start into this folder; mount a volume here to keep them.
ENV FASTEMBED_CACHE_PATH=/models INDEX_DIR=/index
EXPOSE 8021
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8021"]

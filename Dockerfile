FROM python:3.12-slim-bookworm

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src/ /app/src/
COPY ./data/ /app/data/
COPY ./utils/ /app/utils/

EXPOSE 8080

CMD ["streamlit", "run", "src/ui/streamlit_interface.py", "--server.port=8080", "--server.address=0.0.0.0"]
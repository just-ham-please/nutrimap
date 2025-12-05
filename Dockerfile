FROM python:3.12.9
WORKDIR /app


COPY models models
COPY nutrimap_app nutrimap_app
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY data data


RUN pip install --upgrade pip
RUN pip install -e .

#Run container locally
# CMD uvicorn package_folder.api_file:app --reload --host 0.0.0.0

#Run conainer deployed -> GCP
ENV PORT=8080
CMD uvicorn nutrimap_app.api_file:app --reload --host 0.0.0.0 --port $PORT

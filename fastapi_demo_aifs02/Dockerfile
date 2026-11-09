FROM python:3.11-slim

WORKDIR /app

# Install dependencies first to leverage layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code (le modèle n'est PAS dans l'image : il est chargé depuis MLflow au démarrage)
# En dev, ce COPY est masqué par le bind mount -v $(pwd):/app
COPY . .

# The API listens on port 8000
EXPOSE 8000

# Start the server (mode prod, sans --reload). 0.0.0.0 is mandatory so the API
# is reachable from OUTSIDE the container.
# En dev on surcharge ce CMD : docker run ... api-aifs02 uvicorn app:app --host 0.0.0.0 --port 8000 --reload
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Używamy oficjalnego obrazu Python
FROM python:3.11-slim

# Ustawienie katalogu roboczego
WORKDIR /app

# Skopiowanie plików aplikacji
COPY app.py model.pkl preprocessor.pkl requirements.txt ./

# Instalacja zależności
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn

# Otwieramy port 8000
EXPOSE 8000

# Uruchomienie aplikacji
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

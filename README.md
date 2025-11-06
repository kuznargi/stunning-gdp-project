# AstanaGuide AI — RAG система рекомендаций по местам Астаны

Проект предоставляет рекомендации по местам Астаны с использованием Retrieval-Augmented Generation (RAG) и LLM. Frontend - React, Backend - FastAPI.

---

## Возможности
- Семантический поиск POI по тексту + геопозиции
- Фильтр по радиусу, расчёт расстояния и времени пешком
- Проверка «открыто сейчас» по строкам времени работы
- Объяснимые рекомендации: «почему подходит» + план действий
- Поддержка LLM провайдеров: Gemini / OpenAI / Anthropic

---

## Архитектура (высокоуровнево)
1. **Data Processor**: CSV → очистка → эмбеддинги → FAISS индекс  
2. **Retrieval**: семантический поиск + фильтр по расстоянию + ранжирование  
3. **Generation**: LLM формирует 1–3 рекомендации в JSON  
4. **API**: `POST /api/recommendations` → готовый JSON для фронтенда  

---

## Требования
- Python 3.11+
- Node.js + npm
- Зависимости backend: `FastAPI`, `uvicorn`, `sentence-transformers`, `faiss-cpu`, `geopy`, `openai` / `anthropic` / `google-generativeai`
- CSV с POI Астаны (`backend/gis.csv`)

---

## Репозиторий фронтенда
Фронтенд доступен по ссылке:  
[https://github.com/kuznargi/stunning-engine-guide-front.git](https://github.com/kuznargi/stunning-engine-guide-front.git)

---

## Переменные окружения (.env)
Создайте `.env` в корне проекта или `backend/`:

```env
# Gemini (Google)
GOOGLE_API_KEY=AIza...
# или
GEMINI_API_KEY=AIza...

# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

```

## Подготовка данных (однократно)
```
pip install pandas numpy sentence-transformers faiss-cpu geopy python-dotenv
python -m astana_guide_ai.src.data_processor \
  --csv_path "./backend/gis.csv" \
  --output_dir "./astana_guide_ai/data" \
  --model_name "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
  --batch_size 64

```
Артефакты появятся в astana_guide_ai/data/:
processed_pois.json, faiss.index, embeddings.npy, meta.json

## Запуск Backend

Установите зависимости:
```
pip install -r backend/requirements.txt
```

Запустите сервер:
```
PYTHONPATH=backend uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Проверка:
```
curl http://127.0.0.1:8000/health
```
## Запуск Frontend

Клонируйте репозиторий фронтенда:
```
git clone https://github.com/kuznargi/stunning-engine-guide-front.git
cd stunning-engine-guide-front
```

Установите зависимости:
```
npm install

```
Запустите dev-сервер:
```
npm run dev
```

Откройте браузер:
```
http://localhost:8080
```

## Вызов API RAG

Пример без фронтенда:
```
curl -X POST http://127.0.0.1:8000/api/recommendations \
-H 'Content-Type: application/json' \
-d '{
  "query":"Нужно тихое кафе с Wi-Fi",
  "lat":51.1694,
  "lon":71.4491,
  "radius_km":2.0,
  "provider":"gemini",
  "model":"gemini-1.5-flash-latest"
}'
```

Типичные проблемы

404 Gemini → используйте gemini-1.5-flash-latest или обновите google-generativeai

Нет рекомендаций → проверьте наличие processed_pois.json и faiss.index, координаты и radius_km

Проблемы с FAISS/SentenceTransformers → убедитесь в правильной версии Python, переустановите faiss-cpu

## Структура модулей RAG

data_processor.py — обработка CSV и построение индекса

retrieval.py — семантический и геопоиск

generator.py — промпт и вызов LLM

pipeline.py — связка retrieval + generation

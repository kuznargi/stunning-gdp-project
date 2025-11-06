from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from geopy.distance import geodesic

try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")



def _format_distance(distance_km: float) -> str:
    """Возвращает человеко-читаемую строку расстояния и примерного времени пешком.
    Пример: "800м, ~10 мин пешком" или "2.3 км, ~30 мин пешком".
    """
    minutes_walk = int(round((distance_km / 5.0) * 60))  # 5 км/ч — средний шаг
    if distance_km < 1.0:
        meters = int(distance_km * 1000)
        return f"{meters}м, ~{max(3, minutes_walk)} мин пешком"
    else:
        return f"{distance_km:.1f} км, ~{max(5, minutes_walk)} мин пешком"


def _distance_score(distance_km: float, max_distance_km: float) -> float:
    """Нормированный скор близости: 1.0 — возле пользователя, 0.0 — на/за границей max_distance_km."""
    if max_distance_km <= 0:
        return 0.0
    ratio = min(distance_km / max_distance_km, 1.0)
    return max(0.0, 1.0 - ratio)


WEEKDAY_MAP_RU = {
    "пн": 0, "вт": 1, "ср": 2, "чт": 3, "пт": 4, "сб": 5, "вс": 6,
}


@dataclass
class HoursWindow:
    days: Optional[List[int]]
    start: Optional[time]
    end: Optional[time]


def _parse_time(s: str) -> Optional[time]:
    try:
        h, m = s.split(":")
        return time(int(h), int(m))
    except Exception:
        return None


def parse_working_hours(text: str) -> Optional[List[HoursWindow]]:
    """Грубый парсер строк вида:
    - "Ежедневно с 10:00 до 22:00"
    - "Пн-Пт: с 09:00 до 18:00"
    - "Круглосуточно"
    Возвращает список окон работы или None, если распарсить не удалось.
    """
    if not text:
        return None
    t = text.strip().lower()

    if "круглосуточно" in t:
        return [HoursWindow(days=None, start=None, end=None)]

    m = re.search(r"ежедневно\s*с\s*(\d{1,2}:\d{2})\s*до\s*(\d{1,2}:\d{2})", t)
    if m:
        start = _parse_time(m.group(1))
        end = _parse_time(m.group(2))
        if start and end:
            return [HoursWindow(days=None, start=start, end=end)]

    parts = [p.strip() for p in t.split(";") if p.strip()]
    windows: List[HoursWindow] = []
    for p in parts:
        # Пн-Пт: с hh:mm до hh:mm
        m = re.search(r"([а-я]{2})-([а-я]{2}).*?(\d{1,2}:\d{2}).*?(\d{1,2}:\d{2})", p)
        if m:
            d1 = WEEKDAY_MAP_RU.get(m.group(1))
            d2 = WEEKDAY_MAP_RU.get(m.group(2))
            start = _parse_time(m.group(3))
            end = _parse_time(m.group(4))
            if d1 is not None and d2 is not None and start and end:
                if d1 <= d2:
                    days = list(range(d1, d2 + 1))
                else:
                    days = list(range(d1, 7)) + list(range(0, d2 + 1))
                windows.append(HoursWindow(days=days, start=start, end=end))
            continue
        # Отдельный день: Пн: 10:00-19:00
        m = re.search(r"([а-я]{2}).*?(\d{1,2}:\d{2}).*?(\d{1,2}:\d{2})", p)
        if m:
            d = WEEKDAY_MAP_RU.get(m.group(1))
            start = _parse_time(m.group(2))
            end = _parse_time(m.group(3))
            if d is not None and start and end:
                windows.append(HoursWindow(days=[d], start=start, end=end))

    if windows:
        return windows
    return None


def is_open_now(text: str, now: Optional[datetime] = None, tz: str = "Asia/Almaty") -> Optional[bool]:
    """Пытается определить, открыто ли заведение сейчас. Возвращает True/False или None, если неизвестно.
    """
    if not text:
        return None
    windows = parse_working_hours(text)
    if windows is None:
        return None

    if now is None:
        try:
            if ZoneInfo is not None:
                now = datetime.now(ZoneInfo(tz))
            else:
                now = datetime.now()
        except Exception:
            now = datetime.now()

    weekday = now.weekday()  # 0=Mon
    tnow = now.time()

    for w in windows:
        # круглосуточно
        if w.start is None and w.end is None:
            return True
        # по дням
        if w.days is None or weekday in w.days:
            if w.start and w.end:
                if w.start <= w.end:
                    if w.start <= tnow <= w.end:
                        return True
                else:
                    # время через полночь, например 20:00-03:00
                    if tnow >= w.start or tnow <= w.end:
                        return True
    return False


# =============================
# Retrieval Engine
# =============================

class RetrievalEngine:
    """Загружает артефакты, кодирует запросы и выполняет поиск по FAISS."""

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = data_dir
        self._records: Optional[List[Dict[str, Any]]] = None
        self._index = None
        self._model = None
        self._model_name: Optional[str] = None

    @property
    def records(self) -> List[Dict[str, Any]]:
        assert self._records is not None, "Артефакты не загружены"
        return self._records

    def load(self) -> None:
        processed_path = os.path.join(self.data_dir, "processed_pois.json")
        index_path = os.path.join(self.data_dir, "faiss.index")
        meta_path = os.path.join(self.data_dir, "meta.json")

        if not os.path.exists(processed_path) or not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                "Не найдены артефакты данных. Сначала выполните обработку CSV (data_processor.py)."
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            self._model_name = meta.get("model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        with open(processed_path, "r", encoding="utf-8") as f:
            self._records = json.load(f)

        if faiss is None:
            raise ImportError("faiss-cpu не установлен. Установите пакет: pip install faiss-cpu")
        self._index = faiss.read_index(index_path)

        if SentenceTransformer is None:
            raise ImportError("sentence-transformers не установлен. Установите: pip install sentence-transformers")
        self._model = SentenceTransformer(self._model_name)

    def _encode_query(self, text: str) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Модель энкодера не загружена")
        vec = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    def search(self, query: str, top_k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает (scores, indices) формы (1, top_k) по FAISS (cosine similarity via inner product)."""
        if self._index is None:
            raise RuntimeError("FAISS индекс не загружен")
        qvec = self._encode_query(query)
        scores, indices = self._index.search(qvec, top_k)
        return scores[0], indices[0]


_engine_singleton: Optional[RetrievalEngine] = None


def _get_engine() -> RetrievalEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = RetrievalEngine()
        _engine_singleton.load()
    return _engine_singleton


def find_relevant_places(
    user_query: str,
    user_location: Tuple[float, float],  # (lat, lon)
    max_distance_km: float = 5.0,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Находит релевантные места

    Логика:
    1. Векторный поиск по запросу (топ-20)
    2. Фильтрация по расстоянию от пользователя
    3. Проверка времени работы (если указано) — мягкий учет в ранжировании
    4. Ранжирование: final_score = 0.5*semantic_similarity + 0.3*distance_score + 0.2*popularity_score
    5. Возврат топ-N результатов
    """
    if not user_query or not isinstance(user_query, str):
        raise ValueError("user_query должен быть непустой строкой")
    if not (isinstance(user_location, (tuple, list)) and len(user_location) == 2):
        raise ValueError("user_location должен быть кортежем (lat, lon)")

    engine = _get_engine()
    try:
        sim_scores, idxs = engine.search(user_query, top_k=max(20, max_results))
    except Exception as e:
        raise RuntimeError(f"Ошибка векторного поиска: {e}")

    user_lat, user_lon = float(user_location[0]), float(user_location[1])

    candidates: List[Dict[str, Any]] = []
    for score, ix in zip(sim_scores.tolist(), idxs.tolist()):
        if ix < 0:
            continue
        rec = engine.records[ix]
        place_lat = rec.get("lat")
        place_lon = rec.get("lon")
        if place_lat is None or place_lon is None:
            continue
        # Расстояние
        distance_km = geodesic((user_lat, user_lon), (place_lat, place_lon)).kilometers
        if distance_km > max_distance_km:
            continue
        dist_score = _distance_score(distance_km, max_distance_km)

        open_flag = is_open_now(rec.get("working_hours", ""))
        open_bonus = 0.02 if open_flag is True else (-0.02 if open_flag is False else 0.0)

        popularity = float(rec.get("popularity_score", 0.0) or 0.0)

        final_score = 0.5 * float(score) + 0.3 * dist_score + 0.2 * popularity
        final_score = max(0.0, min(1.0, final_score + open_bonus))

        candidates.append({
            "row_id": rec.get("row_id"),
            "id": rec.get("id"),
            "name": rec.get("name"),
            "category": rec.get("category"),
            "subcategory": rec.get("subcategory"),
            "address": rec.get("address"),
            "district": rec.get("district"),
            "city": rec.get("city"),
            "lat": place_lat,
            "lon": place_lon,
            "working_hours": rec.get("working_hours"),
            "instagram": rec.get("instagram"),
            "website": rec.get("website"),
            "phone": rec.get("phone"),
            "distance_km": float(distance_km),
            "distance_text": _format_distance(distance_km),
            "semantic_similarity": float(score),
            "distance_score": float(dist_score),
            "popularity_score": popularity,
            "final_score": float(final_score),
            "open_now": open_flag,
            "description": rec.get("description"),
        })

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates[:max_results]



from __future__ import annotations

import os
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, condecimal
from dotenv import load_dotenv

try:
    from astana_guide_ai.src.pipeline import get_recommendations
except Exception as e:
    raise RuntimeError(
        "Не удалось импортировать пайплайн. Убедитесь, что PYTHONPATH включает корень проекта, "
        "и что артефакты данных подготовлены (см. data_processor.py)."
    ) from e


from app.traffic_simulator import (
    get_current_traffic,
    predict_traffic,
    get_time_coefficient,
    ASTANA_ROADS,
    BRIDGES,
    SUBURBS,
    calculate_eco_impact
)


load_dotenv()
app = FastAPI(title="Kazakhstan Guide AI API", version="0.2.0")

origins = [
    "http://localhost:5173",
    "https://solostack-hackathon.vercel.app",

]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    query: str = Field(..., description="Текстовый запрос пользователя")
    lat: float = Field(..., ge=-90.0, le=90.0, description="Широта пользователя")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Долгота пользователя")
    radius_km: float = Field(3.0, ge=0.1, le=25.0, description="Радиус поиска в км")
    provider: str = Field("openai", description="LLM провайдер: openai|anthropic|gemini")
    model: Optional[str] = Field(None, description="Имя модели (опционально)")
    group_size: Optional[int] = Field(None, ge=2, le=10, description="Количество человек (2-10)")
    group_type: Optional[str] = Field(None, description="Тип группы: family|friends|colleagues|mixed")
    group_preferences: Optional[List[str]] = Field(None, description="Предпочтения: kids_friendly, accessible, budget_friendly")
    language: Optional[str] = Field(None, description="Язык ответа: ru|kk|en (auto-detect если не указан)")


class RecommendationItem(BaseModel):
    name: str
    category: str
    distance: str
    why: str
    action_plan: str
    estimated_time: str
    working_hours: str
    confidence: float
    group_notes: Optional[str] = None
    estimated_cost_per_person: Optional[str] = None
    capacity_suitable: Optional[bool] = None


class RetrievedItem(BaseModel):
    name: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    address: Optional[str] = None
    district: Optional[str] = None
    city: Optional[str] = None
    lat: float
    lon: float
    distance_km: float
    distance_text: str
    working_hours: Optional[str] = None
    instagram: Optional[str] = None
    website: Optional[str] = None
    phone: Optional[str] = None
    open_now: Optional[bool] = None
    popularity_score: Optional[float] = None
    semantic_similarity: Optional[float] = None
    description: Optional[str] = None


class RecommendResponse(BaseModel):
    query: str
    user_location: Dict[str, float]
    radius_km: float
    retrieved: List[RetrievedItem]
    recommendations: List[RecommendationItem]



@lru_cache(maxsize=1)
def load_processed_pois() -> List[Dict[str, Any]]:
    possible_paths = [
        Path(__file__).parent.parent.parent / "astana_guide_ai" / "data" / "processed_pois.json",
        Path("astana_guide_ai/data/processed_pois.json"),
        Path("../astana_guide_ai/data/processed_pois.json"),
    ]

    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break

    if not data_path:
        raise FileNotFoundError(
            f"processed_pois.json not found. Tried paths: {[str(p) for p in possible_paths]}"
        )

    with open(data_path, "r", encoding="utf-8") as f:
        pois = json.load(f)

    return pois


def filter_kazakhstan_locations(
    pois: List[Dict[str, Any]],
    city: Optional[str] = None,
    region: Optional[str] = None,
    min_popularity: float = 0.3,
    limit: int = 50
) -> List[Dict[str, Any]]:
    KZ_LAT_MIN, KZ_LAT_MAX = 40.5, 55.5
    KZ_LON_MIN, KZ_LON_MAX = 46.5, 87.5

    CITY_BOUNDS = {
        "алматы": {"lat": (43.1, 43.4), "lon": (76.7, 77.1)},
        "астана": {"lat": (50.9, 51.4), "lon": (71.0, 71.8)},
        "нур-султан": {"lat": (50.9, 51.4), "lon": (71.0, 71.8)},  # alias
        "шымкент": {"lat": (42.2, 42.4), "lon": (69.4, 69.8)},
        "караганда": {"lat": (49.7, 50.0), "lon": (72.9, 73.3)},
        "актобе": {"lat": (50.2, 50.4), "lon": (57.0, 57.3)},
        "тараз": {"lat": (42.8, 43.0), "lon": (71.2, 71.5)},
        "павлодар": {"lat": (52.2, 52.4), "lon": (76.8, 77.2)},
        "усть-каменогорск": {"lat": (49.9, 50.0), "lon": (82.5, 82.7)},
        "семей": {"lat": (50.3, 50.5), "lon": (80.1, 80.4)},
        "актау": {"lat": (43.5, 43.7), "lon": (51.0, 51.3)},
        "костанай": {"lat": (53.1, 53.3), "lon": (63.5, 63.8)},
        "кызылорда": {"lat": (44.7, 44.9), "lon": (65.4, 65.6)},
        "атырау": {"lat": (47.0, 47.2), "lon": (51.8, 52.0)},
        "петропавловск": {"lat": (54.8, 55.0), "lon": (69.0, 69.3)},
    }

    filtered = []

    for poi in pois:
        lat = poi.get("lat")
        lon = poi.get("lon")
        name = poi.get("name", "").strip()
        popularity = poi.get("popularity_score", 0)
        poi_city = poi.get("city", "").strip()
        poi_region = poi.get("region", "").strip()

        if not name or lat is None or lon is None:
            continue

        if not (KZ_LAT_MIN <= lat <= KZ_LAT_MAX and KZ_LON_MIN <= lon <= KZ_LON_MAX):
            continue

        if city:
            city_lower = city.lower()

            if (poi_city and city_lower in poi_city.lower()) or \
               (poi_region and city_lower in poi_region.lower()):
                pass
            elif city_lower in CITY_BOUNDS:
                bounds = CITY_BOUNDS[city_lower]
                if not (bounds["lat"][0] <= lat <= bounds["lat"][1] and
                        bounds["lon"][0] <= lon <= bounds["lon"][1]):
                    continue
            else:
                continue

        if region and poi_region:
            if region.lower() not in poi_region.lower():
                continue

        if popularity < min_popularity:
            continue

        quality_score = popularity
        if poi.get("address"):
            quality_score += 0.1
        if poi.get("category"):
            quality_score += 0.1
        if poi.get("website") or poi.get("instagram"):
            quality_score += 0.05

        filtered.append({
            "label": name,
            "lat": lat,
            "lon": lon,
            "city": poi_city,
            "region": poi_region,
            "category": poi.get("category", ""),
            "district": poi.get("district", ""),
            "address": poi.get("address", ""),
            "popularity_score": popularity,
            "quality_score": quality_score,
        })

    filtered.sort(key=lambda x: x["quality_score"], reverse=True)

    return filtered[:limit]


# =============================
# API Endpoints
# =============================

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/api/cities")
def cities() -> Dict[str, Any]:
    major_cities = [
        {"name": "Алматы", "name_en": "Almaty", "lat": 43.25, "lon": 76.95, "population": 2000000},
        {"name": "Астана", "name_en": "Astana", "lat": 51.16, "lon": 71.47, "population": 1200000},
        {"name": "Шымкент", "name_en": "Shymkent", "lat": 42.3, "lon": 69.6, "population": 1000000},
        {"name": "Караганда", "name_en": "Karaganda", "lat": 49.8, "lon": 73.1, "population": 500000},
        {"name": "Актобе", "name_en": "Aktobe", "lat": 50.3, "lon": 57.15, "population": 500000},
        {"name": "Тараз", "name_en": "Taraz", "lat": 42.9, "lon": 71.37, "population": 350000},
        {"name": "Павлодар", "name_en": "Pavlodar", "lat": 52.3, "lon": 77.0, "population": 350000},
        {"name": "Усть-Каменогорск", "name_en": "Ust-Kamenogorsk", "lat": 49.95, "lon": 82.6, "population": 300000},
        {"name": "Семей", "name_en": "Semey", "lat": 50.4, "lon": 80.25, "population": 300000},
        {"name": "Актау", "name_en": "Aktau", "lat": 43.6, "lon": 51.15, "population": 250000},
        {"name": "Костанай", "name_en": "Kostanay", "lat": 53.2, "lon": 63.65, "population": 250000},
        {"name": "Кызылорда", "name_en": "Kyzylorda", "lat": 44.8, "lon": 65.5, "population": 250000},
        {"name": "Атырау", "name_en": "Atyrau", "lat": 47.1, "lon": 51.9, "population": 250000},
        {"name": "Петропавловск", "name_en": "Petropavlovsk", "lat": 54.9, "lon": 69.15, "population": 200000},
    ]

    regions = [
        "Алматинская область",
        "Акмолинская область",
        "Актюбинская область",
        "Атырауская область",
        "Восточно-Казахстанская область",
        "Жамбылская область",
        "Западно-Казахстанская область",
        "Карагандинская область",
        "Костанайская область",
        "Кызылординская область",
        "Мангистауская область",
        "Павлодарская область",
        "Северо-Казахстанская область",
        "Туркестанская область",
        "Улытауская область",
        "Абайская область",
        "Жетісуская область",
    ]

    return {
        "cities": major_cities,
        "regions": regions,
        "total_cities": len(major_cities),
        "total_regions": len(regions)
    }


@app.get("/api/locations")
def locations(
    city: Optional[str] = Query(None, description="Filter by city (e.g., 'Алматы', 'Астана')"),
    region: Optional[str] = Query(None, description="Filter by region (e.g., 'Алматинская область')"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of locations to return"),
    min_popularity: float = Query(0.3, ge=0.0, le=1.0, description="Minimum popularity score"),
    category: Optional[str] = Query(None, description="Filter by category (partial match)"),
    include_city_center: bool = Query(True, description="Include city center as first location")
) -> Dict[str, Any]:
    try:
        CITY_CENTERS = {
            "алматы": {"label": "Центр Алматы", "lat": 43.238, "lon": 76.889},
            "астана": {"label": "Центр Астаны (Байтерек)", "lat": 51.1694, "lon": 71.4491},
            "шымкент": {"label": "Центр Шымкента", "lat": 42.3, "lon": 69.6},
            "караганда": {"label": "Центр Караганды", "lat": 49.8, "lon": 73.1},
            "актобе": {"label": "Центр Актобе", "lat": 50.3, "lon": 57.15},
            "тараз": {"label": "Центр Тараза", "lat": 42.9, "lon": 71.37},
            "павлодар": {"label": "Центр Павлодара", "lat": 52.3, "lon": 77.0},
            "усть-каменогорск": {"label": "Центр Усть-Каменогорска", "lat": 49.95, "lon": 82.6},
            "семей": {"label": "Центр Семея", "lat": 50.4, "lon": 80.25},
            "актау": {"label": "Центр Актау", "lat": 43.6, "lon": 51.15},
            "костанай": {"label": "Центр Костаная", "lat": 53.2, "lon": 63.65},
            "кызылорда": {"label": "Центр Кызылорды", "lat": 44.8, "lon": 65.5},
            "атырау": {"label": "Центр Атырау", "lat": 47.1, "lon": 51.9},
            "петропавловск": {"label": "Центр Петропавловска", "lat": 54.9, "lon": 69.15},
        }

        all_pois = load_processed_pois()

        kz_locations = filter_kazakhstan_locations(
            all_pois,
            city=city,
            region=region,
            min_popularity=min_popularity,
            limit=limit * 2  # Get more for category filtering
        )

        if category:
            category_lower = category.lower()
            kz_locations = [
                loc for loc in kz_locations
                if category_lower in loc.get("category", "").lower()
            ]

        if include_city_center and city:
            city_lower = city.lower()
            if city_lower in CITY_CENTERS:
                center = CITY_CENTERS[city_lower]
                center_loc = {
                    "label": center["label"],
                    "lat": center["lat"],
                    "lon": center["lon"],
                    "city": city,
                    "region": "",
                    "category": "Городской центр",
                    "district": "",
                    "address": "",
                    "popularity_score": 1.0
                }
                # Insert at beginning
                kz_locations.insert(0, center_loc)

        kz_locations = kz_locations[:limit]

        for loc in kz_locations:
            loc.pop("quality_score", None)

        return {
            "locations": kz_locations,
            "total": len(kz_locations),
            "city": city,
            "region": region,
            "source": "processed_pois.json"
        }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data file not found: {str(e)}. Please run data_processor.py first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading locations: {str(e)}"
        )


@app.post("/api/recommendations", response_model=RecommendResponse)
def api_recommendations(req: RecommendRequest) -> RecommendResponse:
    provider = (req.provider or "openai").lower()
    if provider not in ("openai", "anthropic", "gemini"):
        raise HTTPException(status_code=400, detail="provider должен быть openai|anthropic|gemini")

    try:
        result = get_recommendations(
            user_query=req.query.strip(),
            user_location=(req.lat, req.lon),
            max_distance_km=float(req.radius_km),
            provider=provider,
            model=req.model,
            group_size=req.group_size,
            group_type=req.group_type,
            group_preferences=req.group_preferences,
            language=req.language,
        )
    except Exception as e:
        # Convert internal errors to 500 with message (avoid leaking stack traces)
        raise HTTPException(status_code=500, detail=str(e))

    # Validate/shape response to the declared schema
    try:
        return RecommendResponse(**result)
    except ValidationError as e:
        # If generator returned empty recommendations, still respond with empty list
        res = {
            "query": result.get("query", req.query),
            "user_location": result.get("user_location", {"lat": req.lat, "lon": req.lon}),
            "radius_km": result.get("radius_km", req.radius_km),
            "retrieved": result.get("retrieved", []),
            "recommendations": result.get("recommendations", []),
        }
        return RecommendResponse(**res)


# =============================
# Smart Astana Traffic Endpoints
# =============================

@app.get("/api/traffic/current")
def api_traffic_current() -> Dict[str, Any]:
    """
    Текущее состояние трафика на всех дорогах, мостах и въездах из пригородов.

    Returns:
        Полные данные о текущем трафике с метриками загрузки
    """
    try:
        traffic_data = get_current_traffic()
        return traffic_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения трафика: {str(e)}")


@app.get("/api/traffic/predictions")
def api_traffic_predictions(
    hours: int = Query(4, ge=1, le=24, description="Количество часов для прогноза (1-24)")
) -> Dict[str, Any]:
    """
    AI-предсказание трафика на N часов вперед.

    Args:
        hours: Количество часов для прогноза (по умолчанию 4)

    Returns:
        Список прогнозов трафика с временными метками
    """
    try:
        predictions = predict_traffic(hours_ahead=hours)
        return {
            "requested_hours": hours,
            "predictions_count": len(predictions),
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания трафика: {str(e)}")


@app.get("/api/roads")
def api_roads_list() -> Dict[str, Any]:
    """
    Список всех мониторимых дорог Астаны с характеристиками.

    Returns:
        Список дорог с детальной информацией
    """
    roads = []
    for road_id, road_data in ASTANA_ROADS.items():
        roads.append({
            "id": road_id,
            "name": road_data["name"],
            "name_en": road_data["name_en"],
            "type": road_data["type"],
            "capacity": road_data["capacity"],
            "speed_limit": road_data["speed_limit"],
            "lanes": road_data["lanes"],
            "length_km": road_data["length_km"],
            "coordinates": road_data["coordinates"],
            "description": road_data["description"]
        })

    return {
        "total": len(roads),
        "roads": roads
    }


@app.get("/api/roads/{road_id}")
def api_road_detail(road_id: str) -> Dict[str, Any]:
    """
    Детальная информация о конкретной дороге с текущим трафиком.

    Args:
        road_id: ID дороги (например, 'kabanbay_batyr')

    Returns:
        Детали дороги + текущий трафик
    """
    if road_id not in ASTANA_ROADS:
        raise HTTPException(status_code=404, detail=f"Дорога '{road_id}' не найдена")

    road_data = ASTANA_ROADS[road_id]

    # Получить текущий трафик
    current_traffic = get_current_traffic()
    road_traffic = next(
        (r for r in current_traffic["roads"] if r["id"] == road_id),
        None
    )

    if not road_traffic:
        raise HTTPException(status_code=500, detail="Не удалось получить трафик для дороги")

    return road_traffic


@app.get("/api/suburbs/all")
def api_suburbs_all() -> Dict[str, Any]:
    """
    Информация о всех пригородах и текущих потоках машин.

    Returns:
        Список пригородов с текущими потоками въезда
    """
    try:
        traffic_data = get_current_traffic()
        return {
            "total": len(traffic_data["suburbs"]),
            "total_daily_inflow": sum(s["daily_inflow"] for s in SUBURBS.values()),
            "suburbs": traffic_data["suburbs"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения данных пригородов: {str(e)}")


@app.get("/api/suburbs/{suburb_id}")
def api_suburb_detail(suburb_id: str) -> Dict[str, Any]:
    """
    Детальная информация о конкретном пригороде.

    Args:
        suburb_id: ID пригорода (например, 'kosshy', 'korgalzhyn')

    Returns:
        Детали пригорода + текущий поток
    """
    if suburb_id not in SUBURBS:
        raise HTTPException(status_code=404, detail=f"Пригород '{suburb_id}' не найден")

    suburb_data = SUBURBS[suburb_id]

    # Получить текущий трафик
    current_traffic = get_current_traffic()
    suburb_traffic = next(
        (s for s in current_traffic["suburbs"] if s["id"] == suburb_id),
        None
    )

    if not suburb_traffic:
        raise HTTPException(status_code=500, detail="Не удалось получить данные пригорода")

    return suburb_traffic


@app.get("/api/bridges/status")
def api_bridges_status() -> Dict[str, Any]:
    """
    Статус всех мостов между левым и правым берегом.

    Returns:
        Список мостов с текущей загрузкой
    """
    try:
        traffic_data = get_current_traffic()
        return {
            "total": len(traffic_data["bridges"]),
            "bridges": traffic_data["bridges"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статуса мостов: {str(e)}")


@app.get("/api/eco/impact")
def api_eco_impact() -> Dict[str, Any]:
    """
    Экологическое воздействие трафика (CO2, расход топлива, экономические потери).

    Returns:
        Метрики экологического воздействия
    """
    try:
        traffic_data = get_current_traffic()
        eco_data = traffic_data.get("eco_impact", {})

        return {
            "timestamp": traffic_data["timestamp"],
            "hour": traffic_data["hour"],
            **eco_data,
            "details": {
                "avg_city_load_percent": traffic_data["avg_city_load_percent"],
                "total_vehicles_on_roads": traffic_data["total_vehicles_on_roads"],
                "message": f"При текущей загрузке {eco_data.get('jam_percentage', 0)}% машин в пробках. "
                          f"Экономические потери составляют ~{eco_data.get('economic_loss_tenge_per_day', 0):,.0f} тенге/день."
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка расчета эко-импакта: {str(e)}")


# Pydantic модели для сравнения маршрутов
class RoutePoint(BaseModel):
    name: str = Field(..., description="Название точки")
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)


class RouteCompareRequest(BaseModel):
    route_a: List[RoutePoint] = Field(..., min_length=2, description="Маршрут A (минимум 2 точки)")
    route_b: List[RoutePoint] = Field(..., min_length=2, description="Маршрут B (минимум 2 точки)")
    departure_time: Optional[str] = Field(None, description="Время отправления (ISO формат)")


@app.post("/api/route/compare")
def api_route_compare(req: RouteCompareRequest) -> Dict[str, Any]:
    """
    Сравнение двух маршрутов с учетом текущего/прогнозируемого трафика.

    Args:
        route_a: Первый маршрут (точки с координатами)
        route_b: Второй маршрут (точки с координатами)
        departure_time: Время отправления (опционально)

    Returns:
        Сравнение маршрутов с рекомендацией
    """
    try:
        # Парсинг времени отправления
        if req.departure_time:
            try:
                # Python 3.11+ поддерживает fromisoformat для большинства ISO форматов
                departure = datetime.fromisoformat(req.departure_time.replace('Z', '+00:00'))
            except ValueError:
                # Fallback: используем dateutil если доступен
                try:
                    from dateutil import parser
                    departure = parser.isoparse(req.departure_time)
                except ImportError:
                    # Если dateutil нет, используем простой парсинг
                    departure = datetime.strptime(req.departure_time[:19], "%Y-%m-%dT%H:%M:%S")
        else:
            departure = datetime.now()

        # Получить трафик на время отправления
        traffic_data = get_current_traffic(departure)

        # Симуляция расчета маршрутов
        # В реальности здесь бы был расчет по графу дорог
        def calculate_route_metrics(route_points: List[RoutePoint], route_name: str):
            # Простая эвристика: расчет расстояния и времени
            total_distance = 0.0
            for i in range(len(route_points) - 1):
                p1, p2 = route_points[i], route_points[i+1]
                # Haversine distance
                R = 6371  # Радиус Земли в км
                lat1, lon1 = math.radians(p1.lat), math.radians(p1.lon)
                lat2, lon2 = math.radians(p2.lat), math.radians(p2.lon)
                dlat, dlon = lat2 - lat1, lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                c = 2 * math.asin(math.sqrt(a))
                total_distance += R * c

            # Средняя скорость зависит от времени суток
            avg_speed = 50.0  # км/ч базовая
            time_coef = get_time_coefficient(departure.hour)

            # Чем выше коэффициент времени, тем ниже скорость
            if time_coef > 1.5:
                avg_speed = 25.0  # пробки
            elif time_coef > 1.0:
                avg_speed = 35.0  # загружено
            elif time_coef < 0.5:
                avg_speed = 70.0  # ночь, свободно

            travel_time_minutes = (total_distance / avg_speed) * 60

            # Оценка расхода топлива
            fuel_per_km = 0.08 if time_coef < 1.0 else 0.12 if time_coef < 1.5 else 0.15
            fuel_consumption = total_distance * fuel_per_km

            return {
                "route_name": route_name,
                "total_distance_km": round(total_distance, 2),
                "estimated_time_minutes": round(travel_time_minutes, 1),
                "estimated_speed_kmh": round(avg_speed, 1),
                "fuel_consumption_liters": round(fuel_consumption, 2),
                "traffic_coefficient": time_coef,
                "waypoints": [{"name": p.name, "lat": p.lat, "lon": p.lon} for p in route_points]
            }

        route_a_metrics = calculate_route_metrics(req.route_a, "Route A")
        route_b_metrics = calculate_route_metrics(req.route_b, "Route B")

        a_time = route_a_metrics["estimated_time_minutes"]
        b_time = route_b_metrics["estimated_time_minutes"]

        if a_time < b_time:
            recommendation = f"Маршрут A быстрее на {round(b_time - a_time, 1)} минут"
            better_route = "route_a"
        elif b_time < a_time:
            recommendation = f"Маршрут B быстрее на {round(a_time - b_time, 1)} минут"
            better_route = "route_b"
        else:
            recommendation = "Оба маршрута примерно одинаковы по времени"
            better_route = "equal"

        # AI-инсайты на основе трафика
        traffic_insights = []
        if traffic_data["avg_city_load_percent"] > 70:
            traffic_insights.append("⚠️ Высокая загруженность города. Рекомендуем отложить поездку или использовать общественный транспорт.")
        if traffic_data["hour"] in range(7, 10):
            traffic_insights.append("🌅 Утренний час-пик. Ожидайте пробки на основных магистралях.")
        elif traffic_data["hour"] in range(17, 20):
            traffic_insights.append("🌆 Вечерний час-пик. Мосты могут быть перегружены.")
        if traffic_data["eco_impact"]["jam_percentage"] > 40:
            traffic_insights.append(f"🌿 Эко-совет: {round(traffic_data['eco_impact']['co2_emissions_tons_per_day'], 1)} тонн CO2 в день. Рассмотрите каршеринг или велосипед.")

        return {
            "departure_time": departure.isoformat(),
            "hour": departure.hour,
            "traffic_condition": "heavy" if traffic_data["avg_city_load_percent"] > 70 else "moderate" if traffic_data["avg_city_load_percent"] > 40 else "free",
            "city_load_percent": traffic_data["avg_city_load_percent"],
            "route_a": route_a_metrics,
            "route_b": route_b_metrics,
            "recommendation": recommendation,
            "better_route": better_route,
            "time_difference_minutes": abs(round(a_time - b_time, 1)),
            "traffic_insights": traffic_insights,
            "eco_impact_snapshot": {
                "co2_kg_per_hour": traffic_data["eco_impact"]["co2_emissions_kg_per_hour"],
                "economic_loss_tenge_per_hour": traffic_data["eco_impact"]["economic_loss_tenge_per_hour"]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сравнения маршрутов: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080)


from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .retrieval import find_relevant_places
from .generator import generate_recommendations


def get_recommendations(
    user_query: str,
    user_location: Tuple[float, float],
    current_time: Optional[str] = None,
    max_distance_km: float = 3.0,
    provider: str = "openai",
    model: Optional[str] = None,
    group_size: Optional[int] = None,
    group_type: Optional[str] = None,
    group_preferences: Optional[List[str]] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Главная функция RAG-системы
    """
    # 1) Retrieval
    places: List[Dict[str, Any]] = find_relevant_places(
        user_query=user_query,
        user_location=user_location,
        max_distance_km=max_distance_km,
        max_results=10,
    )

    # 2) Generation (LLM)
    gen = generate_recommendations(
        user_query=user_query,
        retrieved_places=places,
        provider=provider,
        model=model,
        group_size=group_size,
        group_type=group_type,
        group_preferences=group_preferences,
        language=language,
    )

    return {
        "query": user_query,
        "user_location": {
            "lat": float(user_location[0]),
            "lon": float(user_location[1]),
        },
        "radius_km": float(max_distance_km),
        "retrieved": places,
        "recommendations": gen.get("recommendations", []),
    }


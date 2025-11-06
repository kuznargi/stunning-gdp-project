from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


def detect_language(query: str) -> str:
    cyrillic_chars = sum(1 for c in query if '\u0400' <= c <= '\u04FF')
    latin_chars = sum(1 for c in query if 'a' <= c.lower() <= 'z')

    if cyrillic_chars > latin_chars:
        # Различить русский и казахский по специфичным буквам
        kz_specific = any(c in query for c in 'әіңғүұқөһӘІҢҒҮҰҚӨҺ')
        return 'kk' if kz_specific else 'ru'
    else:
        return 'en'


def build_system_prompt(language: str = 'ru', group_size: Optional[int] = None, group_type: Optional[str] = None) -> str:
    prompts = {
        'ru': (
            "Вы — опытный местный гид Астаны, который помогает людям найти лучшие места поблизости. "
            "Вы даёте 1–3 конкретные рекомендации на основе входного списка релевантных мест и запроса пользователя. "
            "Говорите понятно, дружелюбно и по делу. Не придумывайте факты — если данных нет, честно укажите это. "
            "Учитывайте расстояние, категорию, возможные ограничения по времени работы и предпочтения из запроса. "
            "{group_context}"
            "Отдавайте результат строго в формате JSON согласно схеме."
        ),
        'kk': (
            "Сіз Астана қаласы бойынша тәжірибелі жергілікті гид боласыз, адамдарға жақын маңдағы ең жақсы орындарды табуға көмектесесіз. "
            "Сіз тиісті орындар тізімі мен пайдаланушы сұрауы негізінде 1–3 нақты ұсыныс бересіз. "
            "Түсінікті, достық және іс бойынша сөйлеңіз. Деректер жасамаңыз — егер деректер болмаса, адал айтыңыз. "
            "Қашықтықты, санатты, жұмыс уақыты бойынша шектеулерді және сұраудан келген қалауларды ескеріңіз. "
            "{group_context}"
            "Нәтижені схемаға сәйкес қатаң JSON форматында беріңіз."
        ),
        'en': (
            "You are an experienced local guide in Astana helping people find the best places nearby. "
            "You provide 1–3 specific recommendations based on the input list of relevant places and the user query. "
            "Speak clearly, friendly, and to the point. Don't make up facts — if data is missing, say so honestly. "
            "Consider distance, category, possible working hours restrictions, and preferences from the query. "
            "{group_context}"
            "Return results strictly in JSON format according to the schema."
        )
    }

    group_context = ""
    if group_size and group_size > 1:
        group_contexts = {
            'ru': f"ВАЖНО: Запрос для группы из {group_size} человек (тип: {group_type or 'не указан'}). Рекомендуйте места, подходящие для групп. В поле 'group_notes' укажите советы для компании, а в 'estimated_cost_per_person' — примерную стоимость на человека. ",
            'kk': f"МАҢЫЗДЫ: {group_size} адамнан тұратын топ үшін сұрау (түрі: {group_type or 'көрсетілмеген'}). Топтарға қолайлы орындарды ұсыныңыз. 'group_notes' өрісінде компания үшін кеңестер беріңіз, ал 'estimated_cost_per_person' өрісінде адам басына шамамен құнды көрсетіңіз. ",
            'en': f"IMPORTANT: Request for a group of {group_size} people (type: {group_type or 'not specified'}). Recommend places suitable for groups. In 'group_notes' provide advice for the company, and in 'estimated_cost_per_person' indicate approximate cost per person. "
        }
        group_context = group_contexts.get(language, group_contexts['ru'])

    base_prompt = prompts.get(language, prompts['ru'])
    return base_prompt.format(group_context=group_context)


def _compact_place_item(place: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": place.get("name"),
        "category": place.get("category"),
        "subcategory": place.get("subcategory"),
        "address": place.get("address"),
        "district": place.get("district"),
        "distance": place.get("distance_text"),
        "working_hours": place.get("working_hours") or "Рекомендуем уточнить время работы",
        "website": place.get("website"),
        "instagram": place.get("instagram"),
        "phone": place.get("phone"),
        "open_now": place.get("open_now"),
        "popularity_score": place.get("popularity_score"),
        "description": place.get("description"),
    }


def build_user_prompt(user_query: str, places: List[Dict[str, Any]], language: str = 'ru', group_size: Optional[int] = None, group_preferences: Optional[List[str]] = None) -> str:
    headers = {
        'ru': {
            'input': 'Входные данные',
            'query': 'Запрос пользователя',
            'places': 'Релевантные места (5-10 шт)',
            'subcategory': 'Подкатегория',
            'address': 'Адрес',
            'district': 'Район',
            'distance': 'Расстояние',
            'hours': 'Время работы',
            'description': 'Описание',
            'category_unknown': 'Категория не указана',
            'task': 'Задача',
            'task_points': [
                'Сформируй 1–3 рекомендации (в зависимости от уместности).',
                'Для каждой рекомендации укажи: \'name\', \'category\', \'distance\', \'why\', \'action_plan\', \'estimated_time\', \'working_hours\', \'confidence\'.',
                'Строго следуй JSON-схеме ниже и не добавляй поясняющий текст вне JSON.'
            ],
            'distance_format': 'distance — человеко‑читаемый формат (например, \'800м, ~10 мин пешком\' или \'2.1 км, ~25 мин пешком\').',
            'no_hours': 'Если нет данных по времени работы — запиши \'Рекомендуем уточнить время работы\'.',
            'no_fake': 'Не выдумывай лишних деталей: если чего-то нет в списке мест — не придумывай.'
        },
        'kk': {
            'input': 'Кіріс деректері',
            'query': 'Пайдаланушы сұрауы',
            'places': 'Тиісті орындар (5-10 дана)',
            'subcategory': 'Санат түрі',
            'address': 'Мекенжай',
            'district': 'Аудан',
            'distance': 'Қашықтық',
            'hours': 'Жұмыс уақыты',
            'description': 'Сипаттама',
            'category_unknown': 'Санат көрсетілмеген',
            'task': 'Тапсырма',
            'task_points': [
                '1–3 ұсыныс жасаңыз (орындылыққа байланысты).',
                'Әрбір ұсыныс үшін мыналарды көрсетіңіз: \'name\', \'category\', \'distance\', \'why\', \'action_plan\', \'estimated_time\', \'working_hours\', \'confidence\'.',
                'Төмендегі JSON-схемаға қатаң түрде сәйкес келіңіз және JSON сыртында түсіндірме мәтін қоспаңыз.'
            ],
            'distance_format': 'distance — адамға түсінікті формат (мысалы, \'800м, жаяу ~10 мин\' немесе \'2.1 км, жаяу ~25 мин\').',
            'no_hours': 'Жұмыс уақыты туралы деректер болмаса — \'Жұмыс уақытын нақтылаңыз\' деп жазыңыз.',
            'no_fake': 'Артық мәліметтер ойлап табпаңыз: егер орындар тізімінде жоқ болса — ойлап тапқан жоқ.'
        },
        'en': {
            'input': 'Input data',
            'query': 'User query',
            'places': 'Relevant places (5-10 items)',
            'subcategory': 'Subcategory',
            'address': 'Address',
            'district': 'District',
            'distance': 'Distance',
            'hours': 'Working hours',
            'description': 'Description',
            'category_unknown': 'Category not specified',
            'task': 'Task',
            'task_points': [
                'Form 1–3 recommendations (depending on suitability).',
                'For each recommendation specify: \'name\', \'category\', \'distance\', \'why\', \'action_plan\', \'estimated_time\', \'working_hours\', \'confidence\'.',
                'Strictly follow the JSON schema below and do not add explanatory text outside JSON.'
            ],
            'distance_format': 'distance — human-readable format (e.g., \'800m, ~10 min walk\' or \'2.1 km, ~25 min walk\').',
            'no_hours': 'If no working hours data — write \'Please verify working hours\'.',
            'no_fake': 'Don\'t make up details: if something is not in the list of places — don\'t invent it.'
        }
    }

    h = headers.get(language, headers['ru'])

    header = (
        f"{h['input']}:\n"
        f"{h['query']}: {user_query}\n"
        f"{h['places']}:\n"
    )
    lines = [header]

    if group_preferences:
        prefs_text = {
            'ru': f"Предпочтения группы: {', '.join(group_preferences)}",
            'kk': f"Топ қалаулары: {', '.join(group_preferences)}",
            'en': f"Group preferences: {', '.join(group_preferences)}"
        }
        lines.append(prefs_text.get(language, prefs_text['ru']) + "\n")

    for i, p in enumerate(places, 1):
        pp = _compact_place_item(p)
        part = [
            f"{i}. {pp['name']} ({pp.get('category') or h['category_unknown']})",
        ]
        if pp.get("subcategory"):
            part.append(f"   {h['subcategory']}: {pp['subcategory']}")
        if pp.get("address"):
            part.append(f"   {h['address']}: {pp['address']}")
        if pp.get("district"):
            part.append(f"   {h['district']}: {pp['district']}")
        if pp.get("distance"):
            part.append(f"   {h['distance']}: {pp['distance']}")
        if pp.get("working_hours"):
            part.append(f"   {h['hours']}: {pp['working_hours']}")
        if pp.get("description"):
            desc = str(pp.get("description"))[:240]
            part.append(f"   {h['description']}: {desc}")
        lines.append("\n".join(part))

    lines.append(f"\n{h['task']}:")
    for point in h['task_points']:
        lines.append(f"- {point}")

    # JSON схема с группами
    group_fields = ""
    if group_size and group_size > 1:
        group_fields = ',\n      "group_notes": str | null,\n      "estimated_cost_per_person": str | null,\n      "capacity_suitable": bool | null'

    lines.append(
        "JSON-схема:\n"
        "{\n"
        "  \"recommendations\": [\n"
        "    {\n"
        "      \"name\": str,\n"
        "      \"category\": str,\n"
        "      \"distance\": str,\n"
        "      \"why\": str,\n"
        "      \"action_plan\": str,\n"
        "      \"estimated_time\": str,\n"
        "      \"working_hours\": str,\n"
        "      \"confidence\": float" + group_fields + "\n"
        "    }\n"
        "  ]\n"
        "}\n"
        f"Где: {h['distance_format']}\n"
        f"{h['no_hours']}\n"
        f"{h['no_fake']}\n"
    )

    return "\n".join(lines)


# =============================
# Вызов LLM провайдера
# =============================

class LLMProviderError(RuntimeError):
    pass


def _call_openai(messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> str:

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise LLMProviderError("Пакет openai не установлен: pip install openai>=1.0.0") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMProviderError("Не найден OPENAI_API_KEY в окружении")

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"},  # строгий JSON
        )
        content = resp.choices[0].message.content or "{}"
        return content
    except Exception as e:  # pragma: no cover
        raise LLMProviderError(f"Ошибка вызова OpenAI: {e}") from e


def _call_anthropic(messages: List[Dict[str, str]], model: str = "claude-3-5-sonnet-latest") -> str:
    """Вызывает Anthropic Messages API и старается получить ответ в JSON.
    Возвращает текст ответа.
    """
    try:
        import anthropic  # type: ignore
    except Exception as e:  # pragma: no cover
        raise LLMProviderError("Пакет anthropic не установлен: pip install anthropic") from e

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMProviderError("Не найден ANTHROPIC_API_KEY в окружении")

    client = anthropic.Anthropic(api_key=api_key)
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else None
    user_parts = [m["content"] for m in messages if m["role"] == "user"]

    try:
        resp = client.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": "\n\n".join(user_parts)}],
            temperature=0.3,
            max_tokens=1200,
        )
        # Собираем текстовый контент
        out_text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                out_text += getattr(block, "text", "")
        return out_text.strip() or "{}"
    except Exception as e:  # pragma: no cover
        raise LLMProviderError(f"Ошибка вызова Anthropic: {e}") from e


def _call_gemini(messages: List[Dict[str, str]], model: str = "gemini-1.5-flash") -> str:
    """Вызывает Google Gemini (google-generativeai). Ожидается JSON-строка в ответе.
    Требуется переменная окружения GOOGLE_API_KEY. Установка пакета: pip install google-generativeai
    Проверяет доступность модели через list_models() и выбирает рабочую версию.
    """
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:  # pragma: no cover
        raise LLMProviderError("Пакет google-generativeai не установлен: pip install google-generativeai") from e

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMProviderError("Не найден GOOGLE_API_KEY (или GEMINI_API_KEY) в окружении")

    genai.configure(api_key=api_key)

    # Получаем список доступных моделей с поддержкой generateContent
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                # Убираем префикс "models/" если он есть
                model_name = m.name.replace("models/", "")
                available_models.append(model_name)
    except Exception as e:  # pragma: no cover
        # Если не удалось получить список моделей, используем резервный список
        available_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]

    # Объединяем system и все user-сообщения в один промпт.
    system_text = "\n".join([m["content"] for m in messages if m["role"] == "system"]).strip()
    user_text = "\n\n".join([m["content"] for m in messages if m["role"] == "user"]).strip()

    prompt = (
        (system_text + "\n\n") if system_text else ""
        ) + user_text + "\n\nВыведи строго валидный JSON согласно указанной схеме без пояснений вне JSON."

    # Нормализуем имя запрошенной модели (убираем префикс models/ и суффиксы -latest)
    requested_model = model.replace("models/", "").replace("-latest", "")

    # Порядок попыток моделей: сначала запрошенная, затем доступные, затем резервные
    candidates = [requested_model]

    # Добавляем доступные модели (gemini-1.5-flash в приоритете)
    for am in available_models:
        if am not in candidates:
            if "1.5-flash" in am:
                candidates.insert(1, am)  # flash модели в начало
            else:
                candidates.append(am)

    # Добавляем резервные варианты на всякий случай
    for fallback in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]:
        if fallback not in candidates:
            candidates.append(fallback)

    last_error: Optional[Exception] = None
    tried_models = []

    for m in candidates:
        try:
            # Пробуем без префикса "models/"
            model_obj = genai.GenerativeModel(m)
            resp = model_obj.generate_content(prompt, generation_config={
                "temperature": 0.3,
            })
            text = (getattr(resp, "text", None) or "{}").strip()
            if text:
                return text
        except Exception as e:  # pragma: no cover
            last_error = e
            tried_models.append(m)
            continue

    # Если все попытки провалились — бросаем детальную ошибку
    error_msg = f"Ошибка вызова Gemini. Попробованы модели: {tried_models}. "
    error_msg += f"Доступные модели: {available_models[:5]}. "
    error_msg += f"Последняя ошибка: {last_error}"
    raise LLMProviderError(error_msg)


# =============================
# Публичная функция генерации
# =============================

def generate_recommendations(
    user_query: str,
    retrieved_places: List[Dict[str, Any]],
    provider: str = "openai",  # "openai" | "anthropic" | "gemini"
    model: Optional[str] = None,
    group_size: Optional[int] = None,
    group_type: Optional[str] = None,
    group_preferences: Optional[List[str]] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:

    load_dotenv()

    if not user_query or not isinstance(user_query, str):
        raise ValueError("user_query должен быть непустой строкой")
    if not isinstance(retrieved_places, list) or not retrieved_places:
        return {
            "recommendations": [{
                "name": "Места не найдены",
                "category": "Информация",
                "distance": "—",
                "why": f"В радиусе {max_distance_km if 'max_distance_km' in locals() else 5}км от выбранной точки не найдено подходящих мест.",
                "action_plan": "Попробуйте: 1) Увеличить радиус поиска (передвиньте ползунок до 10км), 2) Выбрать другое 'точное место' ближе к центру города, 3) Изменить запрос (например, убрать слишком специфичные требования).",
                "estimated_time": "—",
                "working_hours": "—",
                "confidence": 0.0
            }]
        }

    detected_language = language if language and language in ('ru', 'kk', 'en') else detect_language(user_query)

    system_prompt = build_system_prompt(language=detected_language, group_size=group_size, group_type=group_type)

    top_places = retrieved_places[:8]

    user_prompt = build_user_prompt(user_query, top_places, language=detected_language, group_size=group_size, group_preferences=group_preferences)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Выбор провайдера
    provider = (provider or "openai").lower()
    if provider == "openai":
        model = model or "gpt-4o-mini"
        raw = _call_openai(messages, model=model)
    elif provider == "anthropic":
        model = model or "claude-3-5-sonnet-latest"
        raw = _call_anthropic(messages, model=model)
    elif provider == "gemini":
        model = model or "gemini-1.5-flash"
        raw = _call_gemini(messages, model=model)
    else:
        raise ValueError("Недопустимый provider. Используйте 'openai', 'anthropic' или 'gemini'.")

    def _parse_json(s: str) -> Dict[str, Any]:
        try:
            return json.loads(s)
        except Exception:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(s[start : end + 1])
                except Exception:
                    pass
            raise

    try:
        data = _parse_json(raw)
    except Exception as e:
        raise LLMProviderError(f"Не удалось распарсить JSON из ответа модели: {e}\nОтвет модели: {raw[:500]}")

    # Минимальная валидация схемы
    if not isinstance(data, dict) or "recommendations" not in data:
        raise LLMProviderError("Ответ модели не соответствует ожидаемой структуре (нет 'recommendations')")

    recs = data.get("recommendations")
    if not isinstance(recs, list):
        raise LLMProviderError("Поле 'recommendations' должно быть списком")

    # Нормализация полей и отсечение до 3
    out_recs: List[Dict[str, Any]] = []
    for r in recs[:3]:
        if not isinstance(r, dict):
            continue
        rec = {
            "name": str(r.get("name", "Без названия")),
            "category": str(r.get("category", "Не указано")),
            "distance": str(r.get("distance", "Не указано")),
            "why": str(r.get("why", "")),
            "action_plan": str(r.get("action_plan", "")),
            "estimated_time": str(r.get("estimated_time", "Не указано")),
            "working_hours": str(r.get("working_hours", "Рекомендуем уточнить время работы")),
            "confidence": float(r.get("confidence", 0.5)),
        }
        if group_size and group_size > 1:
            rec["group_notes"] = r.get("group_notes")
            rec["estimated_cost_per_person"] = r.get("estimated_cost_per_person")
            rec["capacity_suitable"] = r.get("capacity_suitable")
        out_recs.append(rec)

    return {"recommendations": out_recs}


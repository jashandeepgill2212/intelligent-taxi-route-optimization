import os
import re
import math
import hashlib
import logging
from typing import List, Optional, Dict, Any
from collections import OrderedDict

import requests

logger = logging.getLogger(__name__)

NOMINATIM_URL = os.getenv(
    "NOMINATIM_URL",
    "https://nominatim.openstreetmap.org/search"
)
NOMINATIM_USER_AGENT = os.getenv(
    "NOMINATIM_USER_AGENT",
    "TaxiRoute-RL/1.0 (academic project; contact: project@localhost)"
)
# Sensible default viewbox for Punjab / North-West India (min_lon, max_lat, max_lon, min_lat)
NOMINATIM_VIEWBOX = os.getenv(
    "NOMINATIM_VIEWBOX",
    "73.5,32.8,77.5,29.5"
)
REQUEST_TIMEOUT_SEC = int(os.getenv("GEOCODING_TIMEOUT_SEC", "5"))

# Geographic center of the project's operational domain (Ludhiana, Punjab)
LUDHIANA_CENTER_LAT = 30.9010
LUDHIANA_CENTER_LON = 75.8573

# Bounded in-memory LRU cache
_CACHE_MAX = 200
_cache: "OrderedDict[str, Any]" = OrderedDict()

# Authentic verified regional landmarks for the academic project study area
VERIFIED_LOCAL_LANDMARKS: List[Dict[str, Any]] = [
    {
        "name": "CT University",
        "display_name": "CT University, Ferozepur Road, Ludhiana, Punjab 142024, India",
        "latitude": 30.8256,
        "longitude": 75.6421,
        "keywords": ["ct", "ct university", "ct university ludhiana", "ct campus", "ct group"],
    },
    {
        "name": "CT Group of Institutions",
        "display_name": "CT Group of Institutions, Shahpur Campus, Jalandhar, Punjab 144020, India",
        "latitude": 31.2583,
        "longitude": 75.5683,
        "keywords": ["ct", "ct group", "ct institute", "ct group of institutions"],
    },
]

# Keywords indicating explicit international or non-local intent
FOREIGN_LOCATION_KEYWORDS = {
    "usa", "us", "uk", "spain", "france", "italy", "germany", "canada",
    "australia", "america", "europe", "england", "connecticut", "catalonia",
    "catalunya", "catania", "paris", "london", "madrid", "barcelona", "tokyo",
    "california", "new york", "texas", "florida", "rome", "berlin"
}


def _cache_key(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


def _from_cache(key: str) -> Optional[List[Dict[str, Any]]]:
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def _to_cache(key: str, value: List[Dict[str, Any]]) -> None:
    if key in _cache:
        _cache.move_to_end(key)
    else:
        if len(_cache) >= _CACHE_MAX:
            _cache.popitem(last=False)
    _cache[key] = value


def _has_foreign_intent(query: str) -> bool:
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", query.lower()))
    return bool(tokens.intersection(FOREIGN_LOCATION_KEYWORDS))


def _calculate_relevance(
    item: Dict[str, Any],
    query: str,
    raw_item: Optional[Dict[str, Any]] = None
) -> float:
    """
    Score search candidates based on textual match, geographic relevance,
    and regional bias toward Punjab, India.
    """
    q_lower = query.lower().strip()
    q_tokens = [t for t in re.findall(r"\w+", q_lower) if len(t) > 1]
    name_lower = item["name"].lower()
    disp_lower = item["display_name"].lower()

    score = 0.0

    # 1. Textual relevance
    if name_lower == q_lower:
        score += 150.0
    elif name_lower.startswith(q_lower):
        score += 100.0
    elif any(name_part.startswith(q_lower) for name_part in name_lower.split()):
        score += 60.0
    elif q_lower in name_lower:
        score += 40.0
    elif q_lower in disp_lower:
        score += 25.0

    # Token match bonus
    if q_tokens:
        matched_tokens = sum(1 for t in q_tokens if t in disp_lower)
        token_ratio = matched_tokens / len(q_tokens)
        score += token_ratio * 40.0

    # 2. Geographic relevance
    lat = item["latitude"]
    lon = item["longitude"]
    is_foreign_query = _has_foreign_intent(query)

    is_punjab_coord = (29.5 <= lat <= 32.8) and (73.5 <= lon <= 77.5)
    is_ludhiana_coord = (30.6 <= lat <= 31.2) and (75.5 <= lon <= 76.2)

    address = raw_item.get("address", {}) if raw_item else {}
    state = str(address.get("state", "")).lower()
    country = str(address.get("country", "")).lower()
    country_code = str(address.get("country_code", "")).lower()

    is_punjab = is_punjab_coord or ("punjab" in state) or ("punjab" in disp_lower)
    is_ludhiana = is_ludhiana_coord or ("ludhiana" in disp_lower)
    is_india = (country_code == "in") or ("india" in country) or ("india" in disp_lower) or is_punjab

    if not is_foreign_query:
        if is_ludhiana:
            score += 140.0
        elif is_punjab:
            score += 100.0
        elif is_india:
            score += 40.0
        else:
            # Significant demotion for international results when query did not ask for them
            score -= 120.0
    else:
        # If user explicitly asked for international place, reward country/region match
        score += 20.0

    # Importance from Nominatim
    if raw_item and "importance" in raw_item:
        try:
            score += float(raw_item["importance"]) * 10.0
        except (ValueError, TypeError):
            pass

    return score


def search_locations(query: str) -> List[Dict[str, Any]]:
    """
    Search for locations using OpenStreetMap Nominatim with sensible Punjab, India
    viewbox bias, regional prioritization, and in-memory caching.
    """
    query = query.strip()
    if not query:
        raise ValueError("Search query must not be empty.")
    if len(query) < 2:
        raise ValueError("Search query must be at least 2 characters.")
    if len(query) > 200:
        raise ValueError("Search query must not exceed 200 characters.")

    cache_key = _cache_key(query)
    cached = _from_cache(cache_key)
    if cached is not None:
        return cached

    q_lower = query.lower()
    candidates: List[Dict[str, Any]] = []

    # 1. Match against verified regional landmarks (e.g. CT University Ludhiana)
    for landmark in VERIFIED_LOCAL_LANDMARKS:
        l_name = landmark["name"].lower()
        l_disp = landmark["display_name"].lower()
        l_keywords = landmark.get("keywords", [])

        # Check if query matches landmark name, keywords, or all query tokens exist in display name
        q_tokens = [t for t in re.findall(r"\w+", q_lower) if len(t) > 1]
        matches_keyword = any(kw.startswith(q_lower) or q_lower.startswith(kw) or q_lower in kw for kw in l_keywords)
        matches_tokens = bool(q_tokens and all(t in l_disp for t in q_tokens))

        if matches_keyword or matches_tokens or (q_lower in l_name):
            item = {
                "name": landmark["name"],
                "display_name": landmark["display_name"],
                "latitude": landmark["latitude"],
                "longitude": landmark["longitude"],
            }
            item["_score"] = _calculate_relevance(item, query, raw_item={"address": {"state": "Punjab", "country": "India"}})
            candidates.append(item)

    # 2. Query Nominatim with viewbox bias (bounded=0 ensures international queries still work)
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 15,
        "addressdetails": 1,
        "viewbox": NOMINATIM_VIEWBOX,
        "bounded": 0,
    }
    headers = {
        "User-Agent": NOMINATIM_USER_AGENT,
        "Accept-Language": "en",
    }

    try:
        response = requests.get(
            NOMINATIM_URL,
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SEC,
        )
        response.raise_for_status()
        raw_results = response.json()
    except requests.Timeout:
        logger.warning("Nominatim request timed out for query: %s", query)
        raw_results = []
    except requests.RequestException as exc:
        logger.warning("Nominatim request failed: %s", exc)
        raw_results = []
    except ValueError:
        logger.warning("Nominatim returned non-JSON response for query: %s", query)
        raw_results = []

    for item in raw_results:
        try:
            lat = float(item["lat"])
            lon = float(item["lon"])
        except (KeyError, ValueError, TypeError):
            continue

        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue

        display_name: str = item.get("display_name", "")
        name_parts = display_name.split(",")
        name = name_parts[0].strip() if name_parts else display_name

        candidate = {
            "name": name,
            "display_name": display_name,
            "latitude": lat,
            "longitude": lon,
        }
        candidate["_score"] = _calculate_relevance(candidate, query, raw_item=item)
        candidates.append(candidate)

    # Deduplicate candidates (within ~100m proximity or identical display name)
    unique_candidates: List[Dict[str, Any]] = []
    for cand in candidates:
        is_dup = False
        for u in unique_candidates:
            if u["name"] == cand["name"] and abs(u["latitude"] - cand["latitude"]) < 0.001 and abs(u["longitude"] - cand["longitude"]) < 0.001:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(cand)

    # Sort descending by relevance score
    unique_candidates.sort(key=lambda x: x.get("_score", 0.0), reverse=True)

    # Clean internal score key and limit to top 8
    final_results = []
    for c in unique_candidates[:8]:
        final_results.append({
            "name": c["name"],
            "display_name": c["display_name"],
            "latitude": c["latitude"],
            "longitude": c["longitude"],
        })

    _to_cache(cache_key, final_results)
    return final_results

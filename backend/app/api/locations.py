from fastapi import APIRouter, HTTPException, Query
from typing import List

from backend.app.schemas.route_schema import LocationSearchResult
from backend.app.services.geocoding_service import search_locations

router = APIRouter(prefix="/locations", tags=["Geocoding"])


@router.get(
    "/search",
    response_model=List[LocationSearchResult],
    summary="Search for locations by name using Nominatim",
)
def search_location_by_name(
    q: str = Query(
        ...,
        min_length=2,
        max_length=200,
        description="Place name, landmark, address, or station to search for",
    )
):
    """
    Proxy endpoint for Nominatim geocoding.

    Returns up to 8 place suggestions for the given query string.
    Results are cached on the backend to respect Nominatim's usage policy.
    Raises HTTP 422 for invalid queries, HTTP 503 if the geocoding provider
    is unreachable.
    """
    try:
        results = search_locations(q)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Geocoding provider temporarily unavailable. Please try again later.",
        ) from exc

    return results

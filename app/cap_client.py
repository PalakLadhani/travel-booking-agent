"""
HTTP client for the CAP service.
All OData calls live here — tools never call httpx directly.
"""
import os
import httpx
from typing import Optional
from urllib.parse import quote

CAP_BASE_URL = os.getenv("CAP_BASE_URL", "http://localhost:4004/travel")

# 10-second total, 5-second connect. Increase if CAP starts cold.
TIMEOUT = httpx.Timeout(10.0, connect=5.0)


class CAPError(Exception):
    """Raised when CAP returns an error response."""
    pass


def _client() -> httpx.Client:
    """Build a configured httpx client."""
    return httpx.Client(base_url=CAP_BASE_URL, timeout=TIMEOUT)


def list_hotels(city: Optional[str] = None) -> list[dict]:
    """List hotels, optionally filtered by city."""
    url = "/Hotels"
    if city:
        filter_expr = f"city eq '{city}'"
        encoded = quote(filter_expr, safe="'")
        url = f"/Hotels?$filter={encoded}"

    with _client() as client:
        response = client.get(url)

    if response.status_code != 200:
        raise CAPError(f"list_hotels failed: {response.status_code} {response.text}")

    return response.json().get("value", [])
    
def get_hotel(hotel_id: str) -> Optional[dict]:
    """Get a single hotel by ID. Returns None if not found."""
    with _client() as client:
        # OData key syntax: /Hotels(<id>), not /Hotels/<id>
        response = client.get(f"/Hotels({hotel_id})")

    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise CAPError(f"get_hotel failed: {response.status_code} {response.text}")

    return response.json()

def create_booking(payload: dict) -> dict:
    """Create a HOTEL booking via POST /Bookings."""
    payload = {**payload, "bookingType": "hotel"}

    with _client() as client:
        response = client.post(
            "/Bookings",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

    if response.status_code not in (200, 201):
        raise CAPError(f"create_booking failed: {response.status_code} {response.text}")

    return response.json()

# ============================================================
# DESTINATIONS
# ============================================================

def list_destinations(query: Optional[str] = None) -> list[dict]:
    """List destinations, optionally filtered by partial name match (case-insensitive)."""
    url = "/Destinations"
    if query:
        filter_expr = f"contains(tolower(name), tolower('{query}'))"
        encoded = quote(filter_expr, safe="'()")
        url = f"/Destinations?$filter={encoded}"

    with _client() as client:
        response = client.get(url)

    if response.status_code != 200:
        raise CAPError(f"list_destinations failed: {response.status_code} {response.text}")

    return response.json().get("value", [])


def get_destination(destination_id: str) -> Optional[dict]:
    """Get a single destination by ID with its hotels expanded."""
    with _client() as client:
        response = client.get(f"/Destinations({destination_id})?$expand=hotels")

    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise CAPError(f"get_destination failed: {response.status_code} {response.text}")

    return response.json()


def get_destination_by_name(name: str) -> Optional[dict]:
    """Find a destination by exact name (case-insensitive). Includes hotels."""
    filter_expr = f"tolower(name) eq tolower('{name}')"
    encoded = quote(filter_expr, safe="'()")
    url = f"/Destinations?$filter={encoded}&$expand=hotels"

    with _client() as client:
        response = client.get(url)

    if response.status_code != 200:
        raise CAPError(f"get_destination_by_name failed: {response.status_code} {response.text}")

    results = response.json().get("value", [])
    return results[0] if results else None


# ============================================================
# TRAVEL OPTIONS
# ============================================================

def search_travel_options_http(
    from_city: str,
    to_city: str,
    transport_type: Optional[str] = None,
) -> list[dict]:
    """Search travel options between two cities. Sorted by price ascending."""
    parts = [
        f"tolower(fromCity) eq tolower('{from_city}')",
        f"tolower(toCity) eq tolower('{to_city}')",
    ]
    if transport_type:
        parts.append(f"transportType eq '{transport_type.lower()}'")

    filter_expr = " and ".join(parts)
    encoded = quote(filter_expr, safe="'()")
    url = f"/TravelOptions?$filter={encoded}&$orderby=price"

    with _client() as client:
        response = client.get(url)

    if response.status_code != 200:
        raise CAPError(f"search_travel_options failed: {response.status_code} {response.text}")

    return response.json().get("value", [])


def get_travel_option(option_id: str) -> Optional[dict]:
    """Get a single travel option by ID."""
    with _client() as client:
        response = client.get(f"/TravelOptions({option_id})")

    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise CAPError(f"get_travel_option failed: {response.status_code} {response.text}")

    return response.json()


# ============================================================
# TRAVEL BOOKINGS
# ============================================================

def create_travel_booking(payload: dict) -> dict:
    """Create a travel booking via POST /Bookings with bookingType=travel."""
    payload = {**payload, "bookingType": "travel"}

    with _client() as client:
        response = client.post(
            "/Bookings",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

    if response.status_code not in (200, 201):
        raise CAPError(f"create_travel_booking failed: {response.status_code} {response.text}")

    return response.json()
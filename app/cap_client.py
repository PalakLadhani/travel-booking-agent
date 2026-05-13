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
    """Create a booking via POST /Bookings."""
    with _client() as client:
        response = client.post(
            "/Bookings",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

    if response.status_code not in (200, 201):
        raise CAPError(f"create_booking failed: {response.status_code} {response.text}")

    return response.json()
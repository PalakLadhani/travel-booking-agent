"""
Tools the LangGraph agent can call.
All data operations go through the CAP service via cap_client.
"""
from datetime import date
from langchain_core.tools import tool
from app.cap_client import list_hotels, get_hotel, create_booking, CAPError


@tool
def search_hotels(city: str) -> list:
    """Search hotels by city name. Returns a list of hotels matching the city.

    Each hotel includes: ID, name, city, country, pricePerNight, rating, available.
    Use the ID when calling book_hotel.
    """
    try:
        hotels = list_hotels(city=city)
    except CAPError as e:
        return [{"error": f"Search failed: {e}"}]

    if not hotels:
        return [{"error": f"No hotels found in {city}"}]

    # Trim fields — the LLM doesn't need createdAt/modifiedAt etc.
    return [
        {
            "ID": h["ID"],
            "name": h["name"],
            "city": h["city"],
            "country": h.get("country"),
            "pricePerNight": float(h["pricePerNight"]),
            "rating": float(h["rating"]) if h.get("rating") is not None else None,
            "available": h.get("available", True),
        }
        for h in hotels
    ]


@tool
def get_hotel_details(hotel_id: str) -> dict:
    """Get full details for a specific hotel by ID (UUID).

    Returns the hotel record or an error if not found.
    """
    try:
        hotel = get_hotel(hotel_id)
    except CAPError as e:
        return {"error": f"Lookup failed: {e}"}

    if not hotel:
        return {"error": f"Hotel {hotel_id} not found"}

    return {
        "ID": hotel["ID"],
        "name": hotel["name"],
        "city": hotel["city"],
        "country": hotel.get("country"),
        "description": hotel.get("description"),
        "pricePerNight": float(hotel["pricePerNight"]),
        "rating": float(hotel["rating"]) if hotel.get("rating") is not None else None,
        "available": hotel.get("available", True),
    }


@tool
def book_hotel(
    hotel_id: str,
    guest_name: str,
    check_in: str,
    check_out: str,
    number_of_guests: int = 1,
) -> dict:
    """Book a hotel. REQUIRES HUMAN APPROVAL before execution.

    Args:
        hotel_id: UUID of the hotel from search results
        guest_name: Full name of the guest
        check_in: Check-in date in YYYY-MM-DD format
        check_out: Check-out date in YYYY-MM-DD format
        number_of_guests: Number of guests (default 1)

    Returns the created booking with a server-generated UUID.
    """
    # Look up the hotel to validate it exists and compute total price
    try:
        hotel = get_hotel(hotel_id)
    except CAPError as e:
        return {"error": f"Hotel lookup failed: {e}"}

    if not hotel:
        return {"error": f"Hotel {hotel_id} not found"}

    # Validate and parse dates
    try:
        ci = date.fromisoformat(check_in)
        co = date.fromisoformat(check_out)
    except ValueError:
        return {"error": "Dates must be YYYY-MM-DD format"}

    nights = (co - ci).days
    if nights <= 0:
        return {"error": "check_out must be after check_in"}

    total_price = float(hotel["pricePerNight"]) * nights

    # Build OData payload.
    # Associations use the "_ID" suffix — CAP convention.
    # Schema has `hotel : Association to Hotels` → OData field is `hotel_ID`.
    payload = {
        "hotel_ID": hotel_id,
        "guestName": guest_name,
        "checkIn": check_in,
        "checkOut": check_out,
        "numberOfGuests": number_of_guests,
        "totalPrice": total_price,
        "status": "confirmed",
    }

    try:
        booking = create_booking(payload)
    except CAPError as e:
        return {"error": f"Booking failed: {e}"}

    return {
        "status": "confirmed",
        "booking_id": booking["ID"],
        "hotel_name": hotel["name"],
        "guest_name": booking["guestName"],
        "check_in": booking["checkIn"],
        "check_out": booking["checkOut"],
        "number_of_guests": booking["numberOfGuests"],
        "total_price": booking["totalPrice"],
        "nights": nights,
    }


ALL_TOOLS = [search_hotels, get_hotel_details, book_hotel]
"""
Tools the LangGraph agent can call.
All data operations go through the CAP service via cap_client.

There are 6 tools:
- 3 read-only (search_destinations, search_travel_options, search_hotels, get_hotel_details)
- 2 booking tools that require HUMAN APPROVAL via the graph's interrupt mechanism
  (book_travel, book_hotel)
"""
from datetime import date
from langchain_core.tools import tool

from app.cap_client import (
    # destinations
    list_destinations,
    get_destination_by_name,
    # travel
    search_travel_options_http,
    get_travel_option,
    create_travel_booking,
    # hotels
    list_hotels,
    get_hotel,
    create_booking,
    # error type
    CAPError,
)


# ============================================================
# DESTINATIONS — informational, no approval needed
# ============================================================

@tool
def search_destinations(query: str) -> list:
    """Search destinations by partial name match. Returns destinations with
    name, description, best time to visit, attractions, and average hotel price.

    Use this tool when the user mentions a place name (e.g. 'Shimla', 'Goa', 'Jaipur').
    Use this BEFORE searching for travel or hotels — get the destination info first.

    Args:
        query: A partial or full destination name (e.g. "shi" matches "Shimla")
    """
    try:
        results = list_destinations(query=query)
    except CAPError as e:
        return [{"error": f"Destination search failed: {e}"}]

    if not results:
        return [{"error": f"No destinations match '{query}'. Try a different name."}]

    return [
        {
            "ID": d["ID"],
            "name": d["name"],
            "state": d.get("state"),
            "country": d.get("country"),
            "description": d.get("description"),
            "bestTimeToVisit": d.get("bestTimeToVisit"),
            "popularFor": d.get("popularFor"),
            "attractions": d.get("attractions"),
            "averageHotelPrice": float(d["averageHotelPrice"]) if d.get("averageHotelPrice") else None,
        }
        for d in results
    ]


# ============================================================
# TRAVEL OPTIONS — informational, no approval needed
# ============================================================

@tool
def search_travel_options(
    from_city: str,
    to_city: str,
    transport_type: str = None,
) -> list:
    """Search travel options (flights, trains, buses) between two cities,
    sorted by price (cheapest first).

    Use this when the user wants to travel from one place to another.

    Args:
        from_city: Origin city name, e.g. "Delhi", "Mumbai"
        to_city: Destination city name, e.g. "Shimla", "Goa"
        transport_type: Optional filter — "flight", "train", or "bus".
            Omit to see all options.
    """
    try:
        results = search_travel_options_http(from_city, to_city, transport_type)
    except CAPError as e:
        return [{"error": f"Travel search failed: {e}"}]

    if not results:
        return [{"error": f"No travel options found from {from_city} to {to_city}. "
                          "Note: not all routes have direct travel. "
                          "Try nearby cities (e.g. Kalka for Shimla, Bagdogra for Darjeeling)."}]

    return [
        {
            "ID": opt["ID"],
            "transportType": opt["transportType"],
            "provider": opt.get("provider"),
            "routeNumber": opt.get("routeNumber"),
            "fromCity": opt["fromCity"],
            "toCity": opt["toCity"],
            "departureTime": opt.get("departureTime"),
            "arrivalTime": opt.get("arrivalTime"),
            "durationHours": float(opt["durationHours"]) if opt.get("durationHours") else None,
            "price": float(opt["price"]),
            "availableSeats": opt.get("availableSeats"),
            "daysOfWeek": opt.get("daysOfWeek"),
            "classType": opt.get("classType"),
        }
        for opt in results
    ]


# ============================================================
# TRAVEL BOOKING — REQUIRES HUMAN APPROVAL
# ============================================================

@tool
def book_travel(
    travel_option_id: str,
    passenger_name: str,
    travel_date: str,
    number_of_passengers: int = 1,
) -> dict:
    """Book a travel ticket (flight, train, or bus).
    REQUIRES HUMAN APPROVAL before execution.

    Args:
        travel_option_id: UUID of the travel option from search_travel_options
        passenger_name: Full name of the primary passenger
        travel_date: Date of travel in YYYY-MM-DD format
        number_of_passengers: How many passengers (default 1)

    Returns the booking confirmation with a server-generated UUID.
    """
    # Validate the travel option exists
    try:
        option = get_travel_option(travel_option_id)
    except CAPError as e:
        return {"error": f"Travel option lookup failed: {e}"}

    if not option:
        return {"error": f"Travel option {travel_option_id} not found"}

    # Validate date
    try:
        td = date.fromisoformat(travel_date)
    except ValueError:
        return {"error": "travel_date must be in YYYY-MM-DD format"}

    if td < date.today():
        return {"error": "travel_date cannot be in the past"}

    # Compute total price
    total_price = float(option["price"]) * number_of_passengers

    # Build CAP payload
    payload = {
        "travelOption_ID": travel_option_id,
        "guestName": passenger_name,
        "travelDate": travel_date,
        "numberOfPassengers": number_of_passengers,
        "totalPrice": total_price,
        "status": "confirmed",
    }

    try:
        booking = create_travel_booking(payload)
    except CAPError as e:
        return {"error": f"Travel booking failed: {e}"}

    return {
        "status": "confirmed",
        "booking_id": booking["ID"],
        "transport_type": option["transportType"],
        "provider": option.get("provider"),
        "route_number": option.get("routeNumber"),
        "from_city": option["fromCity"],
        "to_city": option["toCity"],
        "departure_time": option.get("departureTime"),
        "travel_date": travel_date,
        "passenger_name": passenger_name,
        "number_of_passengers": number_of_passengers,
        "total_price": total_price,
    }


# ============================================================
# HOTEL TOOLS (existing — unchanged in behavior but kept here for reference)
# ============================================================

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
    """
    try:
        hotel = get_hotel(hotel_id)
    except CAPError as e:
        return {"error": f"Hotel lookup failed: {e}"}

    if not hotel:
        return {"error": f"Hotel {hotel_id} not found"}

    try:
        ci = date.fromisoformat(check_in)
        co = date.fromisoformat(check_out)
    except ValueError:
        return {"error": "Dates must be YYYY-MM-DD format"}

    nights = (co - ci).days
    if nights <= 0:
        return {"error": "check_out must be after check_in"}

    total_price = float(hotel["pricePerNight"]) * nights

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


# ============================================================
# EXPORT
# ============================================================

# Tools that require human approval (the agent.py routing checks this set)
APPROVAL_REQUIRED_TOOLS = {"book_hotel", "book_travel"}

ALL_TOOLS = [
    # Destinations
    search_destinations,
    # Travel
    search_travel_options,
    book_travel,
    # Hotels
    search_hotels,
    get_hotel_details,
    book_hotel,
]
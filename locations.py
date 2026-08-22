# locations.py
#
# Simple Country -> States/UTs lookup for the location-awareness feature.
# Structured as a dict (not a flat list) so adding another country later
# is a one-line addition, not a rewrite. Starting with India only,
# per the project roadmap (Stage 2).

COUNTRIES = {
    "India": [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
        "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
        "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
        "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
        "Uttar Pradesh", "Uttarakhand", "West Bengal",
        "Andaman and Nicobar Islands", "Chandigarh",
        "Dadra and Nagar Haveli and Daman and Diu", "Delhi",
        "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
    ]
}


def get_countries():
    """Return the list of supported countries."""
    return list(COUNTRIES.keys())


def get_states(country: str):
    """Return the list of states/UTs for a given country. Empty list if unknown."""
    return COUNTRIES.get(country, [])


# Approximate (lat, lon) of each state/UT's capital or a major city.
# This is a coarse stand-in for "the location's climate" — one point per
# state, not district-level precision. Good enough for a state-level
# climate lookup; revisit if/when district-level support is added.
STATE_COORDS = {
    "Andhra Pradesh": (16.5062, 80.6480),          # Vijayawada
    "Arunachal Pradesh": (27.0844, 93.6053),        # Itanagar
    "Assam": (26.1445, 91.7362),                    # Guwahati
    "Bihar": (25.5941, 85.1376),                    # Patna
    "Chhattisgarh": (21.2514, 81.6296),             # Raipur
    "Goa": (15.2993, 74.1240),                      # Panaji area
    "Gujarat": (23.0225, 72.5714),                  # Ahmedabad
    "Haryana": (29.0588, 76.0856),                  # Chandigarh area
    "Himachal Pradesh": (31.1048, 77.1734),         # Shimla
    "Jharkhand": (23.3441, 85.3096),                # Ranchi
    "Karnataka": (12.9716, 77.5946),                # Bengaluru
    "Kerala": (8.5241, 76.9366),                    # Thiruvananthapuram
    "Madhya Pradesh": (23.2599, 77.4126),           # Bhopal
    "Maharashtra": (19.0760, 72.8777),              # Mumbai
    "Manipur": (24.8170, 93.9368),                  # Imphal
    "Meghalaya": (25.5788, 91.8933),                # Shillong
    "Mizoram": (23.7271, 92.7176),                  # Aizawl
    "Nagaland": (25.6751, 94.1086),                 # Kohima
    "Odisha": (20.2961, 85.8245),                   # Bhubaneswar
    "Punjab": (30.7333, 76.7794),                   # Chandigarh area
    "Rajasthan": (26.9124, 75.7873),                # Jaipur
    "Sikkim": (27.3389, 88.6065),                   # Gangtok
    "Tamil Nadu": (13.0827, 80.2707),               # Chennai
    "Telangana": (17.3850, 78.4867),                # Hyderabad
    "Tripura": (23.8315, 91.2868),                  # Agartala
    "Uttar Pradesh": (26.8467, 80.9462),            # Lucknow
    "Uttarakhand": (30.3165, 78.0322),              # Dehradun
    "West Bengal": (22.5726, 88.3639),              # Kolkata
    "Andaman and Nicobar Islands": (11.6234, 92.7265),  # Port Blair
    "Chandigarh": (30.7333, 76.7794),
    "Dadra and Nagar Haveli and Daman and Diu": (20.4283, 72.8397),  # Daman
    "Delhi": (28.6139, 77.2090),
    "Jammu and Kashmir": (34.0837, 74.7973),        # Srinagar
    "Ladakh": (34.1526, 77.5771),                   # Leh
    "Lakshadweep": (10.5667, 72.6417),              # Kavaratti
    "Puducherry": (11.9416, 79.8083),
}


def get_coords(state: str):
    """Return (lat, lon) for a state, or None if not found."""
    return STATE_COORDS.get(state)

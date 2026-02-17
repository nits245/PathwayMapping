import math

def flow_to_travel_time(flow: float, distance_km: float) -> float:
    """
    Convert traffic flow (vehicles per hour) and link distance (km) to travel time (seconds).

    Uses a simplified fundamental diagram: flow = -1.4648375 * speed^2 + 93.75 * speed,
    solves the quadratic for speed (free-flow branch), caps to speed limit (60km/h), and adds a fixed
    intersection delay (30s).

    """
    # Constants
    SPEED_LIMIT_KMPH = 60.0           # km/h
    INTERSECTION_DELAY_S = 30.0       # seconds per intersection
    MAX_FLOW = 1500.0                 # vehicle/h (capacity flow)

    # Validate and clamp
    if flow < 0:
        raise ValueError(f"Traffic flow must be non-negative, got {flow}")
    flow = min(flow, MAX_FLOW)

    # Quadratic coefficients: a*v^2 + b*v + c = 0  =>  -1.4648375*s^2 + 93.75*s - flow = 0
    a = -1.4648375
    b = 93.75
    c = -flow
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        discriminant = 0.0
    sqrt_d = math.sqrt(discriminant)

    # Select free-flow branch (higher speed)
    speed = (-b - sqrt_d) / (2 * a)

    # Enforce bounds
    speed = max(speed, 0.0)
    if speed > SPEED_LIMIT_KMPH:
        speed = SPEED_LIMIT_KMPH

    # Handle zero-speed (infinite travel time)
    if speed == 0.0:
        return float('inf')

    # Compute travel time: distance/speed gives hours, convert to seconds and add delay
    travel_time_s = (distance_km / speed) * 3600.0 + INTERSECTION_DELAY_S
    
    return travel_time_s


if __name__ == '__main__':
    # Example usage/test
    distances = [0.5, 1.0, 2.0]
    sample_flows = [0, 200, 500, 1000, 1500]
    for d in distances:
        for f in sample_flows:
            t = flow_to_travel_time(f, d)
            print(f"Flow={f:>4} veh/h, Distance={d:.1f} km -> Time={t:.1f} s")

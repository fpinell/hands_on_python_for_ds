"""Generate a synthetic Milan rentals dataset with non-linear structure.

Compared to rentals_milan.csv, this version (rentals_milan_v2.csv) is
deliberately less linear-friendly: a luxury 3-way interaction, a saturating
size curve, a non-monotone floor effect, and a commute penalty make it so
non-linear models (RandomForest, HistGradientBoosting) can actually win
against a plain LinearRegression on this regression task.

Run from the repo root:
    python scripts/generate_rentals_v2.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
N = 2500
SEED = 42

rng = np.random.default_rng(SEED)


def load_zones() -> pd.DataFrame:
    zones = pd.read_csv(DATA_DIR / "zones_milan.csv")
    zones["neighborhood"] = (
        zones["neighborhood"].astype(str).str.strip().str.lower().str.replace(r"^zone\s+", "", regex=True)
    )
    return zones


def sample_neighborhoods(zones: pd.DataFrame, n: int) -> np.ndarray:
    # central / well-connected zones get a bit more weight; peripheral fewer.
    weights = 1.0 + 0.6 * (zones["transport_score"].values - zones["transport_score"].mean()) / zones["transport_score"].std()
    weights = np.clip(weights, 0.4, 2.0)
    weights = weights / weights.sum()
    return rng.choice(zones["neighborhood"].values, size=n, p=weights)


def sample_sqm(n: int) -> np.ndarray:
    # log-normal-ish, clipped to a realistic range.
    sqm = rng.lognormal(mean=np.log(70), sigma=0.35, size=n)
    return np.clip(sqm, 28, 220).round().astype(int)


def derive_rooms(sqm: np.ndarray) -> np.ndarray:
    # rooms scales with sqm but with noise.
    base = sqm / 25.0 + rng.normal(0, 0.4, size=sqm.size)
    return np.clip(base.round().astype(int), 1, 7)


def derive_bedrooms(rooms: np.ndarray) -> np.ndarray:
    bed = rooms - rng.choice([0, 1], size=rooms.size, p=[0.45, 0.55])
    return np.clip(bed, 1, rooms)


def derive_bathrooms(rooms: np.ndarray) -> np.ndarray:
    # 1 bathroom for small apartments, sometimes 2, rarely 3.
    p2 = np.clip(0.10 + 0.18 * (rooms - 2), 0.05, 0.85)
    out = np.where(rng.random(rooms.size) < p2, 2, 1)
    out = np.where((rooms >= 5) & (rng.random(rooms.size) < 0.25), 3, out)
    return out


def sample_year_built(n: int) -> np.ndarray:
    # bimodal: many pre-1970 + a smaller modern wave.
    a = rng.integers(1900, 1970, size=int(0.65 * n))
    b = rng.integers(1970, 2024, size=n - a.size)
    out = np.concatenate([a, b])
    rng.shuffle(out)
    return out.astype(float)


def sample_floor_raw(n: int, elevator: np.ndarray) -> np.ndarray:
    # Numeric floor 0..7, biased toward 1..3 if no elevator.
    f = rng.integers(0, 8, size=n)
    needs_demote = (~elevator) & (f >= 5) & (rng.random(n) < 0.6)
    f = np.where(needs_demote, rng.integers(0, 5, size=n), f)
    return f


def sample_furnishing(n: int) -> np.ndarray:
    return rng.choice(
        ["furnished", "unfurnished", "partly furnished"],
        size=n,
        p=[0.45, 0.30, 0.25],
    )


def sample_heating(n: int) -> np.ndarray:
    return rng.choice(
        ["autonomous", "central", "heat pump"],
        size=n,
        p=[0.55, 0.35, 0.10],
    )


def sample_energy_class(n: int) -> np.ndarray:
    return rng.choice(
        ["A", "B", "C", "D", "E", "F", "G"],
        size=n,
        p=[0.05, 0.05, 0.10, 0.15, 0.20, 0.20, 0.25],
    )


def sample_metro(n: int, distance_duomo: np.ndarray) -> np.ndarray:
    out = rng.choice(["M1", "M2", "M3", "M5"], size=n, p=[0.28, 0.32, 0.25, 0.15])
    # zones far from the Duomo more often have no nearby metro.
    far = (distance_duomo > 6) & (rng.random(n) < 0.45)
    out = np.where(far, None, out)
    return out


def sample_contract(n: int) -> np.ndarray:
    return rng.choice(
        ["4+4", "3+2", "temporary", "student"],
        size=n,
        p=[0.52, 0.22, 0.16, 0.10],
    )


def saturating_size(sqm: np.ndarray) -> np.ndarray:
    """Piecewise: 10 €/m² up to 110 m², then flatter slope of 4 €/m²."""
    base = np.minimum(sqm, 110)
    extra = np.maximum(0, sqm - 110)
    return 10.0 * base + 4.0 * extra


def floor_effect(floor: np.ndarray, elevator: np.ndarray) -> np.ndarray:
    """Non-monotone in floor; interacts with elevator at the top."""
    out = np.zeros_like(floor, dtype=float)
    out = np.where(floor == 0, -50.0, out)
    out = np.where((floor >= 4) & (floor <= 6), 30.0, out)
    out = np.where((floor >= 7) & elevator, 80.0, out)
    out = np.where((floor >= 7) & (~elevator), -100.0, out)
    return out


def luxury_premium(
    sqm: np.ndarray,
    energy_class: np.ndarray,
    distance_duomo: np.ndarray,
) -> np.ndarray:
    """3-way interaction: big premium for large + efficient + central units."""
    mask = (sqm >= 120) & np.isin(energy_class, ["A", "B"]) & (distance_duomo <= 3.0)
    return np.where(mask, 500.0, 0.0)


def commute_penalty(distance_duomo: np.ndarray, transport_score: np.ndarray) -> np.ndarray:
    mask = (distance_duomo > 5.0) & (transport_score < 5)
    return np.where(mask, -150.0, 0.0)


def age_effect(year_built: np.ndarray) -> np.ndarray:
    """Non-monotone: historic charm, mid-century dip, modern bonus."""
    out = np.zeros_like(year_built, dtype=float)
    out = np.where(year_built < 1920, 60.0, out)
    out = np.where((year_built >= 1920) & (year_built < 1970), -40.0, out)
    out = np.where(year_built >= 2000, 50.0, out)
    return out


def energy_class_linear(ec: np.ndarray) -> np.ndarray:
    table = {"A": 100, "B": 70, "C": 40, "D": 0, "E": -20, "F": -40, "G": -60}
    return np.array([table[c] for c in ec], dtype=float)


def furnishing_bonus(fur: np.ndarray) -> np.ndarray:
    table = {"furnished": 80.0, "partly furnished": 30.0, "unfurnished": 0.0}
    return np.array([table[f] for f in fur], dtype=float)


def heating_bonus(h: np.ndarray) -> np.ndarray:
    table = {"heat pump": 20.0, "autonomous": 0.0, "central": -10.0}
    return np.array([table[v] for v in h], dtype=float)


def contract_multiplier(c: np.ndarray) -> np.ndarray:
    table = {"4+4": 1.00, "3+2": 1.00, "temporary": 1.05, "student": 0.85}
    return np.array([table[v] for v in c], dtype=float)


def messy_price(p: np.ndarray) -> np.ndarray:
    out = np.empty(p.size, dtype=object)
    r = rng.random(p.size)
    for i, val in enumerate(p):
        token = f"{val:.0f}"
        if r[i] < 0.30:
            out[i] = f"€ {token}"
        elif r[i] < 0.45:
            out[i] = f"€{token}"
        else:
            out[i] = token
    return out


def messy_sqm(s: np.ndarray) -> np.ndarray:
    out = np.empty(s.size, dtype=object)
    r = rng.random(s.size)
    for i, val in enumerate(s):
        if r[i] < 0.18:
            out[i] = f"{val} sqm"
        else:
            out[i] = str(val)
    return out


def messy_floor(f: np.ndarray) -> np.ndarray:
    out = np.empty(f.size, dtype=object)
    r = rng.random(f.size)
    for i, val in enumerate(f):
        if val == 0 and r[i] < 0.45:
            out[i] = "ground floor"
        elif val == 1 and r[i] < 0.35:
            out[i] = "first"
        elif r[i] < 0.10:
            out[i] = f"{val}th"
        else:
            out[i] = str(val)
    return out


def messy_furnishing(f: np.ndarray) -> np.ndarray:
    out = np.empty(f.size, dtype=object)
    r = rng.random(f.size)
    for i, val in enumerate(f):
        if val == "partly furnished" and r[i] < 0.15:
            out[i] = rng.choice(["partly", "partly furn."])
        elif val == "unfurnished" and r[i] < 0.10:
            out[i] = rng.choice(["unfurn.", "empty"])
        elif val == "furnished" and r[i] < 0.05:
            out[i] = "furn."
        else:
            out[i] = val
    return out


def messy_heating(h: np.ndarray) -> np.ndarray:
    out = np.empty(h.size, dtype=object)
    r = rng.random(h.size)
    for i, val in enumerate(h):
        if val == "autonomous" and r[i] < 0.15:
            out[i] = rng.choice([" autonomous", "auto.", "autonomous "])
        elif val == "central" and r[i] < 0.10:
            out[i] = "centr."
        elif val == "heat pump" and r[i] < 0.10:
            out[i] = rng.choice(["hp", "heatpump"])
        else:
            out[i] = val
    return out


def messy_energy_class(ec: np.ndarray) -> np.ndarray:
    out = np.empty(ec.size, dtype=object)
    r = rng.random(ec.size)
    for i, val in enumerate(ec):
        if val == "A" and r[i] < 0.7:
            out[i] = rng.choice(["A1", "A2", "A3", "A4"])
        elif r[i] < 0.05:
            out[i] = val.lower()
        else:
            out[i] = val
    # introduce some NaN
    nan_mask = rng.random(ec.size) < 0.25
    out[nan_mask] = np.nan
    return out


def messy_date(n: int) -> np.ndarray:
    months = rng.integers(1, 13, size=n)
    days = rng.integers(1, 28, size=n)
    years = rng.choice([2024, 2025], size=n, p=[0.30, 0.70])
    out = np.empty(n, dtype=object)
    r = rng.random(n)
    for i in range(n):
        if r[i] < 0.55:
            out[i] = f"{years[i]:04d}-{months[i]:02d}-{days[i]:02d}"
        else:
            out[i] = f"{days[i]:02d}/{months[i]:02d}/{years[i]:04d}"
    return out


def messy_neighborhood(nb: np.ndarray) -> np.ndarray:
    out = np.empty(nb.size, dtype=object)
    r = rng.random(nb.size)
    for i, val in enumerate(nb):
        s = str(val).strip()
        if r[i] < 0.30:
            out[i] = s.title()
        elif r[i] < 0.45:
            out[i] = "Zone " + s.title()
        else:
            out[i] = s
    return out


def make_descriptions(bedrooms: np.ndarray, elevator: np.ndarray, year_built: np.ndarray, metro: np.ndarray) -> np.ndarray:
    out = np.empty(bedrooms.size, dtype=object)
    for i in range(bedrooms.size):
        parts = [f"{['One','Two','Three','Four','Five','Six','Seven'][min(bedrooms[i]-1,6)]}-bedroom"]
        if year_built[i] < 1930:
            parts.append("in historic building")
        if elevator[i]:
            parts.append("with elevator")
        if metro[i] is not None and isinstance(metro[i], str):
            parts.append(f"near metro {metro[i]}")
        out[i] = ", ".join(parts) + "."
    return out


def build_price(
    sqm: np.ndarray,
    zone_premium: np.ndarray,
    avg_income: np.ndarray,
    transport_score: np.ndarray,
    distance_duomo: np.ndarray,
    energy_class: np.ndarray,
    floor: np.ndarray,
    elevator: np.ndarray,
    year_built: np.ndarray,
    furnishing: np.ndarray,
    heating: np.ndarray,
    contract: np.ndarray,
    bathrooms: np.ndarray,
) -> np.ndarray:
    """Combine non-linear components into a final price."""
    # Saturating size component, scaled by a soft zone factor.
    size_term = saturating_size(sqm)
    zone_factor = 0.6 + 0.04 * zone_premium  # ~1.0 mid, ~1.6 central, ~0.7 peripheral
    base = size_term * zone_factor

    # Add neighborhood-level structural premia.
    base = base + 0.0012 * (avg_income - avg_income.mean()) * np.minimum(sqm, 110)
    base = base + 12.0 * transport_score

    # Non-linear effects.
    base = base + luxury_premium(sqm, energy_class, distance_duomo)
    base = base + floor_effect(floor, elevator)
    base = base + commute_penalty(distance_duomo, transport_score)
    base = base + age_effect(year_built)
    base = base + energy_class_linear(energy_class)
    base = base + furnishing_bonus(furnishing)
    base = base + heating_bonus(heating)
    base = base + 50.0 * (bathrooms - 1)

    # Multiplicative contract effect.
    base = base * contract_multiplier(contract)

    # Heteroskedastic noise: ~12 % of price.
    noise = rng.normal(0.0, np.maximum(120.0, 0.12 * base))
    price = base + noise

    # Round to a sensible step; clip to a plausible band.
    price = np.clip(price, 380.0, 4500.0)
    return np.round(price).astype(int)


def main() -> None:
    zones = load_zones()
    neighborhood = sample_neighborhoods(zones, N)
    z_lookup = zones.set_index("neighborhood")
    distance_duomo = z_lookup.loc[neighborhood, "distance_duomo_km"].values
    avg_income = z_lookup.loc[neighborhood, "avg_income_eur"].values
    transport_score = z_lookup.loc[neighborhood, "transport_score"].values
    # central zones command a per-m² premium independent of the income trick.
    zone_premium = 22.0 - 1.5 * distance_duomo  # ~22 €/m² central, ~10 peripheral

    sqm = sample_sqm(N)
    rooms = derive_rooms(sqm)
    bedrooms = derive_bedrooms(rooms)
    bathrooms = derive_bathrooms(rooms)
    year_built = sample_year_built(N)
    elevator = rng.random(N) < (0.55 + 0.10 * (year_built > 1970))
    floor = sample_floor_raw(N, elevator)
    furnishing = sample_furnishing(N)
    heating = sample_heating(N)
    energy_class = sample_energy_class(N)
    metro = sample_metro(N, distance_duomo)
    contract = sample_contract(N)

    price = build_price(
        sqm=sqm,
        zone_premium=zone_premium,
        avg_income=avg_income,
        transport_score=transport_score,
        distance_duomo=distance_duomo,
        energy_class=energy_class,
        floor=floor,
        elevator=elevator,
        year_built=year_built,
        furnishing=furnishing,
        heating=heating,
        contract=contract,
        bathrooms=bathrooms,
    )

    # condo fees: function of sqm + central premium + noise.
    condo_fees = 1.5 * sqm + 6.0 * zone_premium + rng.normal(0, 25, size=N)
    condo_fees = np.clip(np.round(condo_fees), 60, 600)
    # ~12% NaN on condo_fees, year_built, and metro
    mask_cf = rng.random(N) < 0.12
    condo_fees_obj = np.where(mask_cf, np.nan, condo_fees)
    mask_yb = rng.random(N) < 0.12
    year_built_obj = np.where(mask_yb, np.nan, year_built)

    deposit_months = rng.choice([1, 2, 3], size=N, p=[0.45, 0.45, 0.10])

    df = pd.DataFrame(
        {
            "id": [f"ML-2025-{i:05d}" for i in rng.integers(1, 99999, size=N)],
            "listing_date": messy_date(N),
            "neighborhood": messy_neighborhood(neighborhood),
            "price": messy_price(price),
            "sqm": messy_sqm(sqm),
            "rooms": rooms,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "floor": messy_floor(floor),
            "elevator": elevator,
            "furnishing": messy_furnishing(furnishing),
            "heating": messy_heating(heating),
            "energy_class": messy_energy_class(energy_class),
            "year_built": year_built_obj,
            "nearest_metro": metro,
            "condo_fees": condo_fees_obj,
            "deposit_months": deposit_months,
            "contract_type": contract,
            "description": make_descriptions(bedrooms, elevator, year_built, metro),
        }
    )

    out_path = DATA_DIR / "rentals_milan_v2.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()

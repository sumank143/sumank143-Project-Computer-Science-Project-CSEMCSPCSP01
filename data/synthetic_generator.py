"""
Synthetic Air Quality Data Generator
=====================================
Generates spatially-correlated time series for PM2.5, NO2, and O3
that mimic the statistical properties of real EEA monitoring data.

Used for pipeline validation before real EEA API integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


# Pollutant-specific baseline statistics (mean, std in µg/m³)
POLLUTANT_STATS = {
    'PM2.5': (25, 12),
    'NO2': (35, 15),
    'O3': (50, 20),
}


def generate_station_coordinates(
    n_stations: int = 10,
    center_lat: float = 48.1,
    center_lon: float = 11.5,
    spread_lat: float = 0.15,
    spread_lon: float = 0.20,
    seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """Generate random monitoring station coordinates around a city center.
    
    Args:
        n_stations: Number of stations to generate.
        center_lat: Central latitude (default: Munich).
        center_lon: Central longitude.
        spread_lat: Latitude spread (degrees).
        spread_lon: Longitude spread (degrees).
        seed: Random seed.
    
    Returns:
        Dictionary mapping station names to (lat, lon) tuples.
    """
    rng = np.random.RandomState(seed)
    coords = {}
    for i in range(n_stations):
        lat = center_lat + rng.uniform(-spread_lat, spread_lat)
        lon = center_lon + rng.uniform(-spread_lon, spread_lon)
        coords[f"ST_{i:02d}"] = (lat, lon)
    return coords


def generate_pollutant_timeseries(
    n_stations: int,
    n_hours: int,
    pollutant: str,
    missing_rate: float = 0.05,
    seed: int = 42
) -> np.ndarray:
    """Generate spatially-correlated air quality time series.
    
    Each station's signal is composed of:
    - A shared regional trend (weather-driven changes)
    - Station-specific diurnal cycle (traffic, heating)
    - Weekly periodicity (weekday vs weekend)
    - Random noise
    - Randomly placed missing values
    
    Args:
        n_stations: Number of stations.
        n_hours: Length of time series in hours.
        pollutant: One of 'PM2.5', 'NO2', 'O3'.
        missing_rate: Fraction of values to set as NaN.
        seed: Random seed.
    
    Returns:
        Array of shape (n_hours, n_stations) with NaN for missing values.
    """
    rng = np.random.RandomState(seed)
    
    if pollutant not in POLLUTANT_STATS:
        raise ValueError(f"Unknown pollutant: {pollutant}. Use one of {list(POLLUTANT_STATS.keys())}")
    
    mean, std = POLLUTANT_STATS[pollutant]
    t = np.arange(n_hours)
    
    # Temporal patterns
    diurnal = np.sin(2 * np.pi * t / 24)
    weekly = 0.3 * np.sin(2 * np.pi * t / 168)
    
    # Shared regional trend (correlated across all stations)
    shared_trend = np.cumsum(rng.randn(n_hours) * 0.3)
    shared_trend = (shared_trend - shared_trend.mean()) / (shared_trend.std() + 1e-8) * std * 0.3
    
    data = np.zeros((n_hours, n_stations))
    for j in range(n_stations):
        amplitude = rng.uniform(0.6, 1.4)
        noise = rng.randn(n_hours) * std * 0.2
        local_diurnal = amplitude * diurnal * std * 0.4
        data[:, j] = mean + local_diurnal + weekly * std + shared_trend + noise
    
    # Concentrations cannot be negative
    data = np.clip(data, 0, None)
    
    # Introduce missing values
    mask = rng.random(data.shape) < missing_rate
    data[mask] = np.nan
    
    return data


def generate_meteorological_data(
    n_hours: int,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """Generate synthetic meteorological co-variates.
    
    Returns:
        Dictionary with keys 'temperature', 'wind_speed', 'humidity'.
    """
    rng = np.random.RandomState(seed + 100)
    t = np.arange(n_hours)
    
    temperature = 15 + 8 * np.sin(2 * np.pi * t / 24 - np.pi / 2) + rng.randn(n_hours) * 2
    wind_speed = np.abs(5 + rng.randn(n_hours) * 2)
    humidity = np.clip(60 + 15 * np.sin(2 * np.pi * t / 24) + rng.randn(n_hours) * 5, 20, 100)
    
    return {
        'temperature': temperature,
        'wind_speed': wind_speed,
        'humidity': humidity,
    }


def generate_full_dataset(
    n_stations: int = 10,
    n_hours: int = 360,
    pollutants: List[str] = None,
    missing_rate: float = 0.05,
    seed: int = 42
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]], Dict[str, np.ndarray]]:
    """Generate a complete synthetic AQ dataset.
    
    Returns:
        Tuple of (aq_data, station_coords, meteo_data).
    """
    if pollutants is None:
        pollutants = ['PM2.5', 'NO2', 'O3']
    
    station_coords = generate_station_coordinates(n_stations, seed=seed)
    
    aq_data = {}
    for i, pol in enumerate(pollutants):
        aq_data[pol] = generate_pollutant_timeseries(
            n_stations, n_hours, pol, missing_rate, seed=seed + i
        )
    
    meteo_data = generate_meteorological_data(n_hours, seed=seed)
    
    return aq_data, station_coords, meteo_data


if __name__ == "__main__":
    aq_data, coords, meteo = generate_full_dataset()
    
    print("=== Synthetic Dataset Summary ===")
    print(f"Stations: {len(coords)}")
    for pol, data in aq_data.items():
        missing_pct = np.isnan(data).mean() * 100
        print(f"{pol}: shape={data.shape}, mean={np.nanmean(data):.1f}, "
              f"std={np.nanstd(data):.1f}, missing={missing_pct:.1f}%")
    print(f"Meteo features: {list(meteo.keys())}")

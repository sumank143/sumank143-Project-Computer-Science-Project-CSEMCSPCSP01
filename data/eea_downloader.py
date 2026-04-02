"""
EEA Air Quality Data Downloader
================================
Downloads hourly air quality measurements from the European Environment
Agency's Air Quality e-Reporting database.

Status: IN PROGRESS — placeholder for Phase 3 integration.

API Reference:
    https://discomap.eea.europa.eu/map/fme/AirQualityExport.htm
"""

import requests
import pandas as pd
from typing import Optional
import time


EEA_BASE_URL = "https://fme.discomap.eea.europa.eu/fmedatastreaming/AirQualityDownload/AQData_Extract.fmw"


def download_eea_data(
    country_code: str = "DE",
    city: str = "Munich",
    pollutant: str = "PM2.5",
    year_from: int = 2023,
    year_to: int = 2024,
    source: str = "E1a",
    output_path: Optional[str] = None,
    max_retries: int = 3
) -> pd.DataFrame:
    """Download hourly AQ data from EEA.
    
    Args:
        country_code: ISO 2-letter country code (e.g., 'DE', 'FR').
        city: City name for filtering.
        pollutant: Pollutant code ('PM2.5', 'NO2', 'O3').
        year_from: Start year.
        year_to: End year.
        source: Data source ('E1a' for verified, 'E2a' for up-to-date).
        output_path: Optional CSV path to save downloaded data.
        max_retries: Number of retry attempts on failure.
    
    Returns:
        DataFrame with columns: datetime, station_id, value, unit.
    
    Raises:
        ConnectionError: If download fails after max_retries.
    
    Note:
        EEA API has rate limits. Use caching for repeated access.
        This function is a placeholder — full implementation in Phase 3.
    """
    # Pollutant code mapping
    pollutant_map = {
        'PM2.5': '6001',
        'PM10': '5',
        'NO2': '8',
        'O3': '7',
        'SO2': '1',
        'CO': '10',
    }
    
    if pollutant not in pollutant_map:
        raise ValueError(f"Unknown pollutant: {pollutant}. "
                        f"Supported: {list(pollutant_map.keys())}")
    
    params = {
        'CountryCode': country_code,
        'CityName': city,
        'Pollutant': pollutant_map[pollutant],
        'Year_from': year_from,
        'Year_to': year_to,
        'Source': source,
        'Output': 'TEXT',
        'TimeCoverage': 'Year',
    }
    
    for attempt in range(max_retries):
        try:
            print(f"Downloading {pollutant} data for {city}, {country_code} "
                  f"({year_from}-{year_to})... (attempt {attempt + 1})")
            
            response = requests.get(EEA_BASE_URL, params=params, timeout=120)
            response.raise_for_status()
            
            # Parse CSV response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            if output_path:
                df.to_csv(output_path, index=False)
                print(f"Saved to {output_path}")
            
            print(f"Downloaded {len(df)} records.")
            return df
            
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 5
                print(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise ConnectionError(
                    f"Failed to download after {max_retries} attempts: {e}"
                )


if __name__ == "__main__":
    print("EEA Downloader — Phase 3 placeholder")
    print("Full implementation will support:")
    print("  - Bulk download with caching")
    print("  - Multiple cities and pollutants")
    print("  - Automatic station metadata retrieval")
    print("  - Rate limit handling with exponential backoff")

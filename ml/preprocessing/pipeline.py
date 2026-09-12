import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional


class TaxiDataPipeline:
    """
    Configurable preprocessing pipeline for historical & synthetic taxi trip datasets.
    Supports automatic column mapping, outlier filtering, temporal feature extraction,
    and traffic congestion index computation.
    """

    COLUMN_MAPS = {
        'pickup_datetime': ['pickup_datetime', 'tpep_pickup_datetime', 'lpep_pickup_datetime', 'pickup_time'],
        'dropoff_datetime': ['dropoff_datetime', 'tpep_dropoff_datetime', 'lpep_dropoff_datetime', 'dropoff_time'],
        'pickup_lat': ['pickup_latitude', 'pickup_lat', 'Start_Lat'],
        'pickup_lng': ['pickup_longitude', 'pickup_lng', 'Start_Lng'],
        'dropoff_lat': ['dropoff_latitude', 'dropoff_lat', 'End_Lat'],
        'dropoff_lng': ['dropoff_longitude', 'dropoff_lng', 'End_Lng'],
        'trip_distance': ['trip_distance', 'distance', 'distance_km'],
        'passenger_count': ['passenger_count', 'passengers']
    }

    def __init__(self, min_lat: float = -90.0, max_lat: float = 90.0,
                 min_lng: float = -180.0, max_lng: float = 180.0,
                 max_speed_kmh: float = 150.0, max_distance_km: float = 100.0):
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lng = min_lng
        self.max_lng = max_lng
        self.max_speed_kmh = max_speed_kmh
        self.max_distance_km = max_distance_km

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and standardize column names based on candidate aliases."""
        renames = {}
        for target, aliases in self.COLUMN_MAPS.items():
            for alias in aliases:
                if alias in df.columns and target not in renames.values():
                    renames[alias] = target
                    break
        df = df.rename(columns=renames)
        return df

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute complete data preprocessing pipeline.
        Returns cleaned DataFrame and statistical report.
        """
        initial_rows = len(df)
        df = self.standardize_columns(df)

        # Drop rows with null essential columns if present
        essential = [col for col in ['pickup_lat', 'pickup_lng', 'dropoff_lat', 'dropoff_lng'] if col in df.columns]
        if essential:
            df = df.dropna(subset=essential)

        # Coordinate filtering
        if 'pickup_lat' in df.columns:
            df = df[(df['pickup_lat'] >= self.min_lat) & (df['pickup_lat'] <= self.max_lat) &
                    (df['pickup_lng'] >= self.min_lng) & (df['pickup_lng'] <= self.max_lng)]
        if 'dropoff_lat' in df.columns:
            df = df[(df['dropoff_lat'] >= self.min_lat) & (df['dropoff_lat'] <= self.max_lat) &
                    (df['dropoff_lng'] >= self.min_lng) & (df['dropoff_lng'] <= self.max_lng)]

        # Datetime processing
        if 'pickup_datetime' in df.columns:
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
            df = df.dropna(subset=['pickup_datetime'])
            df['hour'] = df['pickup_datetime'].dt.hour
            df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
            # Rush hour: Weekdays (0-4) during 8-10 AM or 5-7 PM (vectorized)
            is_weekday = df['day_of_week'] < 5
            morning_rush = (df['hour'] >= 8) & (df['hour'] <= 10)
            evening_rush = (df['hour'] >= 17) & (df['hour'] <= 19)
            df['is_rush_hour'] = (is_weekday & (morning_rush | evening_rush)).astype(int)

        if 'dropoff_datetime' in df.columns and 'pickup_datetime' in df.columns:
            df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], errors='coerce')
            df['trip_duration_min'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60.0
            df = df[df['trip_duration_min'] > 0.5]  # At least 30 seconds

        # Distance & Speed calculations
        if 'trip_distance' in df.columns:
            df = df[(df['trip_distance'] > 0.1) & (df['trip_distance'] <= self.max_distance_km)]

        if 'trip_distance' in df.columns and 'trip_duration_min' in df.columns:
            df['speed_kmh'] = (df['trip_distance'] / (df['trip_duration_min'] / 60.0))
            df = df[(df['speed_kmh'] > 1.0) & (df['speed_kmh'] <= self.max_speed_kmh)]
            # Congestion index: lower speed relative to free flow (e.g. 40 km/h) means high congestion
            df['congestion_index'] = np.clip(1.0 + (40.0 - df['speed_kmh']) / 40.0, 1.0, 3.0)

        cleaned_rows = len(df)
        report = {
            'initial_rows': initial_rows,
            'cleaned_rows': cleaned_rows,
            'removed_rows': initial_rows - cleaned_rows,
            'retention_rate': round(cleaned_rows / max(initial_rows, 1) * 100, 2)
        }

        return df, report


def load_and_preprocess_dataset(filepath: str, output_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Helper to load raw CSV/Parquet dataset and save cleaned version."""
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)

    pipeline = TaxiDataPipeline()
    cleaned_df, report = pipeline.preprocess(df)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cleaned_df.to_csv(output_path, index=False)

    return cleaned_df, report

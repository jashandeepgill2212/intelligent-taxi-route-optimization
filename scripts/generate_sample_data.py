import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.preprocessing.pipeline import TaxiDataPipeline


def generate_synthetic_taxi_data(num_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic taxi trips for testing and offline fallback.
    Default area centered around Ludhiana / Urban City area.
    """
    np.random.seed(seed)
    base_lat, base_lng = 30.9010, 75.8573  # Center coordinates
    
    start_time = datetime(2026, 9, 1, 6, 0, 0)
    data = []
    
    for i in range(num_samples):
        # Random offsets within ~10km radius
        p_lat = base_lat + np.random.uniform(-0.05, 0.05)
        p_lng = base_lng + np.random.uniform(-0.05, 0.05)
        d_lat = base_lat + np.random.uniform(-0.05, 0.05)
        d_lng = base_lng + np.random.uniform(-0.05, 0.05)
        
        # Approximate Euclidean distance in km (1 deg lat ~ 111km)
        lat_dist = (d_lat - p_lat) * 111.0
        lng_dist = (d_lng - p_lng) * 111.0 * np.cos(np.radians(base_lat))
        trip_distance = float(np.sqrt(lat_dist**2 + lng_dist**2)) + np.random.uniform(0.2, 1.5)
        
        pickup_dt = start_time + timedelta(minutes=int(np.random.uniform(0, 1440)))
        
        # Speed varies by hour (rush hour slower)
        hour = pickup_dt.hour
        is_rush = 1 if (8 <= hour <= 10 or 17 <= hour <= 19) else 0
        avg_speed = np.random.uniform(15.0, 25.0) if is_rush else np.random.uniform(30.0, 45.0)
        
        trip_duration = (trip_distance / avg_speed) * 60.0  # minutes
        dropoff_dt = pickup_dt + timedelta(minutes=trip_duration)
        passenger_count = int(np.random.choice([1, 1, 1, 2, 3, 4], p=[0.6, 0.15, 0.1, 0.08, 0.05, 0.02]))
        
        data.append({
            'pickup_datetime': pickup_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'dropoff_datetime': dropoff_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'pickup_latitude': round(p_lat, 6),
            'pickup_longitude': round(p_lng, 6),
            'dropoff_latitude': round(d_lat, 6),
            'dropoff_longitude': round(d_lng, 6),
            'trip_distance': round(trip_distance, 2),
            'passenger_count': passenger_count
        })
        
    return pd.DataFrame(data)


def main():
    raw_dir = os.path.abspath('ml/data/raw')
    proc_dir = os.path.abspath('ml/data/processed')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    
    raw_path = os.path.join(raw_dir, 'sample_trips.csv')
    proc_path = os.path.join(proc_dir, 'cleaned_trips.csv')
    
    print(f"Generating synthetic taxi dataset with 500 samples...")
    raw_df = generate_synthetic_taxi_data(num_samples=500)
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved raw dataset to: {raw_path}")
    
    print("Running preprocessing pipeline...")
    pipeline = TaxiDataPipeline()
    cleaned_df, report = pipeline.preprocess(raw_df)
    cleaned_df.to_csv(proc_path, index=False)
    
    print(f"Saved cleaned dataset to: {proc_path}")
    print(f"Preprocessing Summary: {report}")


if __name__ == '__main__':
    main()

# Chapter 8: Dataset & Data Preprocessing

## 8.1 Dataset Strategy & Fallback
The platform supports historical taxi trip datasets (such as NYC Yellow Taxi Trip records or urban GPS traces) containing fields: `pickup_datetime`, `dropoff_datetime`, `pickup_latitude`, `pickup_longitude`, `dropoff_latitude`, `dropoff_longitude`, `trip_distance`, `passenger_count`.

## 8.2 Data Cleaning & Feature Engineering
- **Coordinate Validation**: Filters out GPS anomalies (latitude outside $[-90, 90]$, longitude outside $[-180, 180]$).
- **Impossible Trip Removal**: Drops trips with zero distance, duration $< 30$ seconds, or speed $> 150$ km/h.
- **Temporal Feature Extraction**: Derives hour, day of week, and binary rush-hour flag (weekdays 8-10 AM and 5-7 PM).
- **Congestion Index Calculation**: Computes speed ratios to estimate localized traffic congestion indices.

'use client';

import React, { useEffect, useState } from 'react';
import { LocationPoint, RouteCoordinate, RouteResponse } from '../types';

interface MapComponentProps {
  pickup: LocationPoint;
  destination: LocationPoint;
  onSelectPickup: (loc: LocationPoint) => void;
  onSelectDestination: (loc: LocationPoint) => void;
  routes?: Record<string, RouteResponse>;
  recommended_algorithm?: string;
}

export default function MapComponent({
  pickup,
  destination,
  onSelectPickup,
  onSelectDestination,
  routes,
  recommended_algorithm
}: MapComponentProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="w-full h-[500px] bg-slate-900 rounded-xl flex items-center justify-center text-slate-400 border border-slate-800 animate-pulse">
        <span>Loading Interactive Leaflet Map...</span>
      </div>
    );
  }

  // Client-side Leaflet Container
  const LeafletContainer = dynamicLeafletMap(pickup, destination, onSelectPickup, onSelectDestination, routes, recommended_algorithm);

  return <div className="w-full h-[500px] rounded-xl overflow-hidden shadow-2xl border border-slate-800">{LeafletContainer}</div>;
}

function dynamicLeafletMap(
  pickup: LocationPoint,
  destination: LocationPoint,
  onSelectPickup: (loc: LocationPoint) => void,
  onSelectDestination: (loc: LocationPoint) => void,
  routes?: Record<string, RouteResponse>,
  recommended_algorithm?: string
) {
  const L = require('leaflet');
  require('leaflet/dist/leaflet.css');

  // Fix default icon urls in Leaflet
  delete L.Icon.Default.prototype._getIconUrl;
  L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
    iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
    shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  });

  const { MapContainer, TileLayer, Marker, Popup, Polyline, useMapEvents } = require('react-leaflet');

  function LocationSelector() {
    const [clickMode, setClickMode] = useState<'pickup' | 'dest'>('pickup');
    useMapEvents({
      click(e: any) {
        const coords = { lat: parseFloat(e.latlng.lat.toFixed(4)), lng: parseFloat(e.latlng.lng.toFixed(4)) };
        if (clickMode === 'pickup') {
          onSelectPickup(coords);
          setClickMode('dest');
        } else {
          onSelectDestination(coords);
          setClickMode('pickup');
        }
      },
    });
    return null;
  }

  // Algorithm color mapping
  const algorithmColors: Record<string, string> = {
    'Dijkstra': '#3b82f6', // Blue
    'A*': '#eab308',       // Yellow
    'DQN': '#10b981',      // Green
  };

  const center: [number, number] = [(pickup.lat + destination.lat) / 2, (pickup.lng + destination.lng) / 2];

  return (
    <MapContainer center={center} zoom={13} style={{ height: '100%', width: '100%' }} scrollWheelZoom={true}>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <LocationSelector />

      {/* Pickup Marker */}
      <Marker position={[pickup.lat, pickup.lng]}>
        <Popup>
          <div className="text-slate-900 font-bold">
            🚕 Pickup Location
            <br />
            Lat: {pickup.lat}, Lng: {pickup.lng}
          </div>
        </Popup>
      </Marker>

      {/* Destination Marker */}
      <Marker position={[destination.lat, destination.lng]}>
        <Popup>
          <div className="text-slate-900 font-bold">
            🏁 Destination Location
            <br />
            Lat: {destination.lat}, Lng: {destination.lng}
          </div>
        </Popup>
      </Marker>

      {/* Render Polylines for each algorithm route */}
      {routes &&
        Object.entries(routes).map(([algName, route]) => {
          if (!route.success || !route.route_coordinates.length) return null;
          const positions: [number, number][] = route.route_coordinates.map((c) => [c.lat, c.lng]);
          const isRecommended = algName === recommended_algorithm;

          return (
            <Polyline
              key={algName}
              positions={positions}
              pathOptions={{
                color: algorithmColors[algName] || '#0284c7',
                weight: isRecommended ? 6 : 3,
                opacity: isRecommended ? 0.9 : 0.6,
                dashArray: algName === 'A*' ? '8, 8' : undefined
              }}
            >
              <Popup>
                <div className="text-slate-900 font-semibold">
                  {isRecommended ? '⭐ Recommended Route: ' : ''}{algName}
                  <br />
                  Distance: {route.total_distance_km} km
                  <br />
                  ETA: {route.total_time_min} mins
                </div>
              </Popup>
            </Polyline>
          );
        })}
    </MapContainer>
  );
}

import React, { useEffect, useState, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet-draw/dist/leaflet.draw.css";
import "leaflet.heat";
import { h3 } from "h3-js";
import $ from "jquery";
import "jquery-ui/ui/widgets/autocomplete";
import "jquery-ui/themes/base/all.css";

const MapComponent = () => {
  const [taxaNames, setTaxaNames] = useState([]);
  const mapRef = useRef(null);
  const [map, setMap] = useState(null);
  const [heatLayer] = useState(L.layerGroup());
  const [polygonLayer] = useState(L.layerGroup());
  const [annotationPolygonLayer] = useState(L.layerGroup());
  const [hexagonLayer] = useState(L.layerGroup());
  const [drawnItems] = useState(new L.FeatureGroup());

  useEffect(() => {
    // Fetch taxa names from the JSON file and initialize auto-suggest
    fetch("/static/taxa_names.json")
      .then((response) => response.json())
      .then((data) => {
        setTaxaNames(data);
        // Initialize jQuery UI autocomplete
        $("#taxa_name").autocomplete({
          source: data,
          minLength: 7,
          change: function (event, ui) {
            // Check if the value is in the list
            if (ui.item == null) {
              // If not, clear the input field
              $(this).val("");
              alert("Please select a valid taxa name from the list.");
            }
          },
        });
      })
      .catch((error) => console.error("Error loading taxa names:", error));
  }, []);

  useEffect(() => {
    const initialMap = L.map("map").setView([39, 34], 3);
    setMap(initialMap);

    // Add different tile layers
    const worldLightGrayBase = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 16 }
    ).addTo(initialMap);

    const worldImagery = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 16 }
    );

    const worldTopoMap = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 16 }
    );

    const openStreetMap = L.tileLayer(
      "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      {
        attribution: "&copy; OpenStreetMap contributors",
        maxZoom: 19,
      }
    );

    // Add layer control
    const baseMaps = {
      "World Light Gray Base": worldLightGrayBase,
      "World Imagery": worldImagery,
      "World Topo Map": worldTopoMap,
      "OpenStreetMap": openStreetMap,
    };

    // Add layer control with overlays
    const overlays = {
      "Prediction (Heat Map)": heatLayer,
      "Prediction (Polygons)": polygonLayer,
      "Prediction (Hexagons)": hexagonLayer,
      "Saved Annotation": annotationPolygonLayer,
      "Current Annotation": drawnItems,
    };

    L.control.layers(baseMaps, overlays).addTo(initialMap);

    // Initialize Leaflet Draw Control
    const drawControl = new L.Control.Draw({
      draw: {
        polygon: true,
        polyline: false,
        rectangle: true,
        circle: false,
        marker: false,
        circlemarker: false,
      },
      edit: {
        featureGroup: drawnItems,
        remove: true, // Enable removal of drawn shapes
      },
    });
    initialMap.addControl(drawControl);

    // Handle creation of new drawn items
    initialMap.on(L.Draw.Event.CREATED, function (event) {
      const layer = event.layer;
      drawnItems.addLayer(layer);
    });
  }, [heatLayer, polygonLayer, annotationPolygonLayer, hexagonLayer, drawnItems]);

  const createHeatMap = (coordinates) => {
    heatLayer.clearLayers();
    const newHeatLayer = L.heatLayer(coordinates, { radius: 25, minOpacity: 0.5 });
    heatLayer.addLayer(newHeatLayer);
  };

  const createPolygons = (hullPoints) => {
    polygonLayer.clearLayers();
    hullPoints.forEach((points) => {
      L.polygon(points, {
        color: "red",
        weight: 2,
        fill: true,
        fillColor: "yellow",
        fillOpacity: 0.4,
      })
        .bindPopup("Prediction")
        .addTo(polygonLayer);
    });
  };

  const createAnnotationPolygons = (hullPoints) => {
    annotationPolygonLayer.clearLayers();
    hullPoints.forEach((points) => {
      L.polygon(points, {
        color: "green",
        weight: 2,
        fill: true,
        fillColor: "green",
        fillOpacity: 0.4,
      })
        .bindPopup("Saved Annotation")
        .addTo(annotationPolygonLayer);
    });
  };

  const createHexagons = (hullPoints, hexResolution) => {
    hexagonLayer.clearLayers();
    hullPoints.forEach((h_points) => {
      const hexagons = h3.polyfill(
        [h_points.map((point) => [point[1], point[0]])], // Convert to [longitude, latitude]
        hexResolution
      );

      hexagons.forEach((hexId) => {
        const hexBoundary = h3.h3ToGeoBoundary(hexId);
        const hexCoords = hexBoundary.map((point) => [point[1], point[0]]); // Convert back to [latitude, longitude]
        L.polygon(hexCoords, {
          color: "green",
          weight: 1,
          fill: true,
          fillColor: "green",
          fillOpacity: 0.4,
        }).addTo(hexagonLayer);
      });
    });
  };

  const generatePrediction = async (
    taxa_name,
    resolution,
    threshold,
    model,
    disable_ocean_mask
  ) => {
    return fetch("/generate_prediction", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        taxa_name,
        resolution: parseInt(resolution),
        threshold: parseFloat(threshold),
        model,
        disable_ocean_mask,
      }),
    });
  };

  const handleGeneratePrediction = () => {
    const taxa_name = document.getElementById("taxa_name").value;
    const resolution = document.getElementById("resolution").value;
    const threshold = document.getElementById("threshold").value;
    const model = document.getElementById("model").value;
    const disable_ocean_mask = document.getElementById("disable_ocean_mask").checked;

    drawnItems.clearLayers();

    generatePrediction(taxa_name, resolution, threshold, model, disable_ocean_mask)
      .then((response) => response.json())
      .then((data) => {
        map.setView(data.preds_center, 4);
        createHeatMap(data.coordinates);
        createPolygons(data.hull_points);
        createHexagons(data.hull_points, parseInt(resolution));
        createAnnotationPolygons(data.saved_annotation);
      });
  };

  const saveAnnotation = async (polygons, taxa_name) => {
    return fetch("/save_annotation", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        taxa_name,
        polygons,
      }),
    });
  };

  const handleSaveAnnotation = () => {
    const polygons = [];
    drawnItems.eachLayer((layer) => {
      const coordinates = layer
        .getLatLngs()[0]
        .map((latlng) => [latlng.lat, latlng.lng]);
      polygons.push(coordinates);
    });

    if (polygons.length === 0) {
      alert("Please draw an annotation to save it.");
    } else {
      const taxa_name = document.getElementById("taxa_name").value;
      saveAnnotation(polygons, taxa_name)
        .then((response) => response.json())
        .then((data) => {
          console.log("Polygons saved successfully!");
          drawnItems.clearLayers();
          annotationPolygonLayer.clearLayers();
          createAnnotationPolygons(data.polygons);
        })
        .catch((error) => {
          console.error("Error saving polygons:", error);
          alert("Failed to save polygons. Please try again.");
        });
    }
  };

  const loadAnnotation = async (taxa_name) => {
    return fetch("/load_annotation", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        taxa_name,
      }),
    });
  };

  const handleLoadAnnotation = () => {
    const taxa_name = document.getElementById("taxa_name").value;
    loadAnnotation(taxa_name)
      .then((response) => response.json())
      .then((data) => {
        drawnItems.clearLayers();
        createAnnotationPolygons(data.polygons);
        console.log("Polygons loaded successfully:", data.polygons);
      })
      .catch((error) => {
        console.error("Error loading polygons:", error);
        alert("Error loading polygons:", error);
      });
  };

  const handleClearAnnotation = () => {
    const taxa_name = document.getElementById("taxa_name").value;
    saveAnnotation([], taxa_name)
      .then((response) => {
        drawnItems.clearLayers();
        annotationPolygonLayer.clearLayers();
        console.log("Polygons cleared successfully");
      })
      .catch((error) => {
        console.error("Error clearing polygons:", error);
        alert("Error clearing polygons:", error);
      });
  };

  return (
    <div>
      <input id="taxa_name" type="text" placeholder="Enter taxa name" />
      <input id="resolution" type="text" placeholder="Resolution" />
      <input id="threshold" type="text" placeholder="Threshold" />
      <input id="model" type="text" placeholder="Model" />
      <input id="disable_ocean_mask" type="checkbox" />
      <button id="generate_prediction" onClick={handleGeneratePrediction}>
        Generate Prediction
      </button>
      <button id="save_annotation" onClick={handleSaveAnnotation}>
        Save Annotation
      </button>
      <button id="load_annotation" onClick={handleLoadAnnotation}>
        Load Annotation
      </button>
      <button id="clear_annotation" onClick={handleClearAnnotation}>
        Clear Annotation
      </button>
      <div id="map" style={{ height: "600px" }} ref={mapRef}></div>
    </div>
  );
};

export default MapComponent;

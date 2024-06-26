let taxaNames = [];

// Fetch taxa names from the JSON file and initialize auto-suggest
fetch('/static/taxa_names.json')
    .then(response => response.json())
    .then(data => {
        taxaNames = data;
        // Initialize jQuery UI autocomplete
        $("#taxa_name").autocomplete({
            source: taxaNames,
            minLength: 7,
            change: function(event, ui) {
                // Check if the value is in the list
                if (ui.item == null) {
                    // If not, clear the input field
                    $(this).val('');
                    alert("Please select a valid taxa name from the list.");
                }
            }
        });
    })
    .catch(error => console.error('Error loading taxa names:', error));


// Initialize the map
const map = L.map("map").setView([39, 34], 3);

// Add different tile layers
const worldLightGrayBase = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
  maxZoom: 16
}).addTo(map);

const worldImagery = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
  maxZoom: 16
});

const worldTopoMap = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', {
  maxZoom: 16
});

const openStreetMap = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors',
  maxZoom: 19
});

// Add layer control
const baseMaps = {
  "World Light Gray Base": worldLightGrayBase,
  "World Imagery": worldImagery,
  "World Topo Map": worldTopoMap,
  "OpenStreetMap": openStreetMap
};

// Initialize an empty layer group for the heatmap
const heatLayer = L.layerGroup();

// Initialize an empty layer group for the polygons
const polygonLayer = L.layerGroup().addTo(map);

// Initialize an empty layer group for the annotations
const annotationPolygonLayer = L.layerGroup().addTo(map);

// Initialize an empty layer group for the hexagons
const hexagonLayer = L.layerGroup();

// Initialize FeatureGroup to store drawn items
const drawnItems = new L.FeatureGroup().addTo(map);

// Add layer control with overlays
const overlays = {
  "Prediction (Heat Map)": heatLayer,
  "Prediction (Polygons)": polygonLayer,
  "Prediction (Hexagons)": hexagonLayer,
  "Saved Annotation": annotationPolygonLayer,
  "Current Annotation": drawnItems,
};

L.control.layers(baseMaps, overlays).addTo(map);

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
    remove: true, // Disable removal of drawn shapes
  },
});
map.addControl(drawControl);

// Handle creation of new drawn items
map.on(L.Draw.Event.CREATED, function (event) {
  const layer = event.layer;
  drawnItems.addLayer(layer);
});

// Function to create heatmap
function createHeatMap(coordinates) {
  // Clear previous heatmap layers
  heatLayer.clearLayers();
  const newHeatLayer = L.heatLayer(coordinates, { radius: 25, minOpacity: 0.5 });
  heatLayer.addLayer(newHeatLayer);
}

// Function to create polygons
function createPolygons(hullPoints) {
  // Clear previous polygon layers
  polygonLayer.clearLayers();
  hullPoints.forEach(points => {
    L.polygon(points, { color: 'red', weight: 2, fill: true, fillColor: 'yellow', fillOpacity: 0.4 })
      .bindPopup('Prediction')
      .addTo(polygonLayer);
  });
}

// Function to load annotations
function createAnnotationPolygons(hullPoints) {
  // Clear previous polygon layers
  annotationPolygonLayer.clearLayers();
  hullPoints.forEach(points => {
    L.polygon(points, { color: 'green', weight: 2, fill: true, fillColor: 'green', fillOpacity: 0.4 })
      .bindPopup('Saved Annotation')
      .addTo(annotationPolygonLayer);
  });
}

// FIXME !!!

// Function to create hexagons within polygons
function createHexagons(hullPoints, hexResolution) {
  // Clear previous hexagon layers
  hexagonLayer.clearLayers();
  hullPoints.forEach(h_points => {
    // Generate hexagons within the polygon
    const hexagons = h3.polyfill(
      [h_points.map(point => [point[1], point[0]])], // Convert to [longitude, latitude]
      hexResolution
    );

    // Convert hexagons to Leaflet polygons
    hexagons.forEach(hexId => {
      const hexBoundary = h3.h3ToGeoBoundary(hexId);
      const hexCoords = hexBoundary.map(point => [point[1], point[0]]); // Convert back to [latitude, longitude]
      L.polygon(hexCoords, { color: 'green', weight: 1, fill: true, fillColor: 'green', fillOpacity: 0.4 }).addTo(hexagonLayer);
    });
  });
}

// predict button functionality
document.getElementById("generate_prediction").onclick = () => {
  const taxa_name = document.getElementById("taxa_name").value;
  const resolution = document.getElementById("resolution").value;
  const threshold = document.getElementById("threshold").value;
  const model = document.getElementById("model").value;
  const disable_ocean_mask = document.getElementById("disable_ocean_mask").checked;

  // Clear previous drawn items
  drawnItems.clearLayers();

  generatePrediction(taxa_name, resolution, threshold, model, disable_ocean_mask)
    .then((response) => response.json())
    .then((data) => {

      // Center map on the predicted area
      map.setView(data.preds_center, 4);

      // Create heatmap with the returned coordinates
      createHeatMap(data.coordinates);

      // Create polygons with the returned hull points
      createPolygons(data.hull_points);

      // Create hexagons within the polygons
      createHexagons(data.hull_points, parseInt(resolution));

      // Add saved annotation
      createAnnotationPolygons(data.saved_annotation);

      
    });
};

async function generatePrediction(taxa_name, resolution, threshold, model, disable_ocean_mask) {
  return fetch("/generate_prediction", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      taxa_name: taxa_name,
      resolution: parseInt(resolution),
      threshold: parseFloat(threshold),
      model: model,
      disable_ocean_mask: disable_ocean_mask,
    }),
  });
}


// Save annotation button functionality
document.getElementById("save_annotation").onclick = () => {
  const polygons = [];
  drawnItems.eachLayer((layer) => {
    const coordinates = layer.getLatLngs()[0].map((latlng) => [latlng.lat, latlng.lng]);
    polygons.push(coordinates); 
  });

  if ( polygons.length == 0 ) {
    alert('Please draw an annotation to save it.')
  } else {
    // Send polygons data to backend
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
  };
};

async function saveAnnotation(polygons, taxa_name) {
  return fetch("/save_annotation", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      taxa_name: taxa_name,
      polygons: polygons,
    }),
  });
}


// Load annotation button functionality
document.getElementById("load_annotation").onclick = () => {
  // Load polygons from DB
  const taxa_name = document.getElementById("taxa_name").value;
  loadAnnotation(taxa_name)
    .then((response) => response.json())
    .then((data) => {
      drawnItems.clearLayers();
      console.log(data.polygons)
      createAnnotationPolygons(data.polygons);
      console.log("Polygons load successfully:", data.polygons);
    })
    .catch((error) => {
      console.error("Error loading polygons:", error);
      alert("Error loading polygons:", error);
    });
};

async function loadAnnotation(taxa_name) {
  return fetch("/load_annotation", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      taxa_name: taxa_name,
    }),
  });
}


// Clear annotation button functionality
document.getElementById("clear_annotation").onclick = () => {
  // Clear polygons from DB
  const taxa_name = document.getElementById("taxa_name").value;
  // to clear existing annotation save empty annotation
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

import { colorToWhiteScale } from "./util.js";

const MAPS = {
  "gray-world-canvas": L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    {
      attribution: "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
      maxZoom:4,
    }
  ),
  "world-imagery": L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    {
      attribution:
        "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
    }
  ),
  "topo-map": L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    {
      attribution:
        "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), and the GIS User Community",
    }
  ),
};

const map = L.map("map").setView([42, -72], 4);

var currentMap = MAPS["topo-map"];
currentMap.addTo(map);

const mapSelector = document.getElementById("map-options");
mapSelector.addEventListener("change", (event) => {
  map.removeLayer(currentMap);
  currentMap = MAPS[event.target.value];
  currentMap.addTo(map);
});

async function renderMap() {
  annLayer.clearLayers();
  const resolution = document.getElementById("resolution").value;
  const k = document.getElementById("k").value;
  const center = map.getCenter();
  const lat = document.getElementById("lat").value;
  const lon = document.getElementById("lon").value;
  fetchHexagons(lat, lon, resolution, k)
    .then((response) => response.json())
    .then((data) => {
      data.hexagons.forEach((element) => {
        const poly = L.polygon(element["boundary"], {
          color: colorToWhiteScale(element["p"]),
          fill: true,
        });

        poly.addEventListener("click", (e) => {
          if (poly.options.color === "#ff0000") {
            poly.setStyle({ color: colorToWhiteScale(element["p"]) });
            console.log("Reset: ", element["h3_id"]);
          } else {
            poly.setStyle({ color: "#ff0000" });
            console.log("Mark: ", element["h3_id"]);
          }
        });

        poly.addTo(annLayer);
      });
    });
}

var annLayer = L.layerGroup().addTo(map);
document
  .getElementById("lat")
  .addEventListener("change", async (e) => await renderMap());
document
  .getElementById("lon")
  .addEventListener("change", async (e) => await renderMap());
document
  .getElementById("resolution")
  .addEventListener("change", async (e) => await renderMap());
document
  .getElementById("k")
  .addEventListener("change", async (e) => await renderMap());

document.getElementById("generate").onclick = () => {
  annLayer.clearLayers();
  const resolution = document.getElementById("resolution").value;
  const k = document.getElementById("k").value;
  const center = map.getCenter();
  const lat = document.getElementById("lat").value;
  const lon = document.getElementById("lon").value;
  fetchHexagons(lat, lon, resolution, k)
    .then((response) => response.json())
    .then((data) => {
      data.hexagons.forEach((element) => {
        const poly = L.polygon(element["boundary"], {
          color: colorToWhiteScale(element["p"]),
          fill: true,
        });

        poly.addEventListener("click", (e) => {
          if (poly.options.color === "#ff0000") {
            poly.setStyle({ color: colorToWhiteScale(element["p"]) });
            console.log("Reset: ", element["h3_id"]);
          } else {
            poly.setStyle({ color: "#ff0000" });
            console.log("Mark: ", element["h3_id"]);
          }
        });

        poly.addTo(annLayer);
      });
    });
};

// async function fetchHexagons(lat, lon, resolution, k) {
//   return fetch("/generate_random_k_ring", {
//     method: "POST",
//     headers: {
//       "Content-Type": "application/json",
//     },
//     body: JSON.stringify({
//       k: parseInt(k),
//       coord: {
//         lat: parseInt(lat),
//         lon: parseInt(lon),
//       },
//       resolution: parseInt(resolution),
//     }),
//   });
// };

async function fetchHexagons(resolution) {
  return fetch("/generate_all_hexagons", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      resolution: parseInt(resolution),
    }),
  });
}


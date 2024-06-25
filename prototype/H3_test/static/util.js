export function colorToWhiteScale(value) {
  value = Math.max(0, Math.min(1, value));
  // const full = { r: 126, g: 5, b: 247 };
  const full = { r: 0, g: 0, b: 0 }; // Define default hexagon color
  const white = { r: 255, g: 255, b: 255 }; 
  const r = full.r + (white.r - full.r) * value;
  const g = full.g + (white.g - full.g) * value;
  const b = full.b + (white.b - full.b) * value;
  const toHex = (x) => {
    const hex = Math.round(x).toString(16);
    return hex.length === 1 ? "0" + hex : hex;
  };

  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

'use strict';

function initialWindowState(saved, area) {
  const fallbackWidth = Math.min(1440, area.width - 48);
  const fallbackHeight = Math.min(940, area.height - 48);
  const minWidth = Math.min(1024, area.width);
  const minHeight = Math.min(640, area.height);
  const finite = (value, fallback) => Number.isFinite(value) ? Math.round(value) : fallback;
  const width = Math.max(minWidth, Math.min(area.width, finite(saved?.width, fallbackWidth)));
  const height = Math.max(minHeight, Math.min(area.height, finite(saved?.height, fallbackHeight)));
  const x = Math.min(area.x + area.width - width, Math.max(area.x, finite(saved?.x, area.x + (area.width - width) / 2)));
  const y = Math.min(area.y + area.height - height, Math.max(area.y, finite(saved?.y, area.y + (area.height - height) / 2)));
  return { x, y, width, height, minWidth, minHeight, maximized: saved?.maximized === true };
}

// Quit requests must first let the renderer veto close for unsaved edits. Only
// an actually closed window permits stopping its backend; Cancel keeps it alive.
function requestQuit({ event, window, shutdown, alreadyQuitting }) {
  if (alreadyQuitting) return;
  event.preventDefault();
  if (window && !window.isDestroyed()) window.close();
  else shutdown();
}

module.exports = { initialWindowState, requestQuit };

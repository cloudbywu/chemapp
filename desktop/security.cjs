'use strict';

const DESKTOP_HEADER = 'X-ChemApp-Desktop-Token';

function sameOrigin(value, origin) {
  try {
    const url = new URL(value);
    return url.protocol === 'http:' && url.origin === origin && !url.username && !url.password;
  } catch {
    return false;
  }
}

function allowedNavigation(value, origin) {
  if (!sameOrigin(value, origin)) return false;
  const url = new URL(value);
  return url.pathname === '/' || url.pathname === '/index.html';
}

function allowedLoadingRequest(details, loadingURL, webContentsId) {
  return details.webContentsId === webContentsId && details.resourceType === 'mainFrame' && details.url === loadingURL;
}

function allowedRequest(details, origin, webContentsId) {
  if (details.webContentsId !== webContentsId) return false;
  if (details.resourceType === 'subFrame' || details.resourceType === 'webSocket') return false;
  if (sameOrigin(details.url, origin)) return true;
  // Plotly exports and browser download blobs stay in this document's origin.
  return details.url.startsWith(`blob:${origin}/`) || details.url.startsWith('data:image/');
}

function requestHeaders(details, origin, webContentsId, secrets) {
  if (!allowedRequest(details, origin, webContentsId)) return null;
  const headers = { ...details.requestHeaders };
  // Strip both ours and case variants supplied by renderer code before injecting.
  for (const key of Object.keys(headers)) {
    if (['x-chemapp-desktop-token', 'x-chemapp-access-token', 'x-chemapp-admin-token'].includes(key.toLowerCase())) {
      delete headers[key];
    }
  }
  if (sameOrigin(details.url, origin)) {
    headers[DESKTOP_HEADER] = secrets.desktop;
    headers['X-ChemApp-Access-Token'] = secrets.access;
    headers['X-ChemApp-Admin-Token'] = secrets.admin;
  }
  return headers;
}

module.exports = { DESKTOP_HEADER, allowedLoadingRequest, sameOrigin, allowedNavigation, allowedRequest, requestHeaders };

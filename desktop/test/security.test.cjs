'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { allowedLoadingRequest, sameOrigin, allowedNavigation, allowedRequest, requestHeaders } = require('../security.cjs');
const origin = 'http://127.0.0.1:45678';
const request = { url: `${origin}/api/spectra`, webContentsId: 7, resourceType: 'xhr', requestHeaders: {} };
const secrets = { desktop: 'desktop-secret', access: 'access-secret', admin: 'admin-secret' };

test('origin validation uses URL origin, not prefix matching', () => {
  assert(sameOrigin(`${origin}/api`, origin));
  for (const url of [`${origin}.evil.test/`, 'http://127.0.0.1:45679/', 'https://127.0.0.1:45678/', 'http://user@127.0.0.1:45678/', 'file:///tmp/page', 'not-a-url']) {
    assert.equal(sameOrigin(url, origin), false, url);
  }
});
test('navigation only permits own entrypoint, denying API pages and remote destinations', () => {
  assert(allowedNavigation(`${origin}/#plot`, origin));
  assert(allowedNavigation(`${origin}/index.html`, origin));
  for (const suffix of ['/docs', '/api/reports', '/other.html']) assert.equal(allowedNavigation(origin + suffix, origin), false);
  assert.equal(allowedNavigation('https://electronjs.org', origin), false);
});
test('request allowlist binds document identity and blocks frames and external network', () => {
  assert(allowedRequest(request, origin, 7));
  for (const modified of [{webContentsId: 8}, {webContentsId: undefined}, {resourceType: 'subFrame'}, {resourceType: 'webSocket'}, {url: 'https://example.org/upload'}, {url: 'file:///etc/passwd'}]) {
    assert.equal(allowedRequest({...request, ...modified}, origin, 7), false);
  }
  assert(allowedRequest({...request, url: `blob:${origin}/uuid`}, origin, 7));
  assert.equal(allowedRequest({...request, url: 'blob:https://evil.test/uuid'}, origin, 7), false);
});
test('credentials only attach to exact owned origin and replace header case variants', () => {
  const headers = requestHeaders({...request, requestHeaders: {'x-chemapp-desktop-token':'spoof', 'X-CHEMAPP-ADMIN-TOKEN':'spoof', 'X-ChemApp-Reviewer-Token':'reviewer'}}, origin, 7, secrets);
  assert.equal(headers['X-ChemApp-Desktop-Token'], secrets.desktop);
  assert.equal(headers['X-ChemApp-Admin-Token'], secrets.admin);
  assert.equal(headers['X-ChemApp-Reviewer-Token'], 'reviewer');
  assert.equal(headers['x-chemapp-desktop-token'], undefined);
  assert.equal(requestHeaders({...request, url: 'https://evil.test'}, origin, 7, secrets), null);
  assert.deepEqual(requestHeaders({...request, url: `blob:${origin}/uuid`}, origin, 7, secrets), {});
});

test('startup document is allowed only for the exact local file and owned main frame', () => {
  const loading = 'file:///opt/ChemApp/loading.html';
  const request = {url:loading, webContentsId:7, resourceType:'mainFrame'};
  assert(allowedLoadingRequest(request,loading,7));
  for (const override of [{url:loading+'?x'}, {url:'file:///etc/passwd'}, {webContentsId:8}, {resourceType:'subFrame'}, {resourceType:'xhr'}]) {
    assert.equal(allowedLoadingRequest({...request,...override},loading,7),false);
  }
});

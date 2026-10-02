'use strict';
const { app, BrowserWindow, dialog, Menu, session, screen } = require('electron');
const fs = require('node:fs');
const path = require('node:path');
const { randomUUID } = require('node:crypto');
const { pathToFileURL } = require('node:url');
const { captureWorkspace } = require('./capture-workspace.cjs');
const { initialWindowState, requestQuit } = require('./window-state.cjs');
const { Backend, launchPaths } = require('./backend.cjs');
const { allowedNavigation, allowedRequest, allowedLoadingRequest, requestHeaders, sameOrigin } = require('./security.cjs');

app.setName('ChemApp');
// No sandbox disabling switches, Node-enabled renderer, preload, or IPC bridge.
app.enableSandbox();
let window;
let backend;
let ready = false;
let quitting = false;
let failureShowing = false;
let logPath;
let capturing = false;

function log(message) {
  let clean = String(message);
  for (const value of Object.values(backend?.secrets || {})) clean = clean.replaceAll(value, '[redacted]');
  try {
    if (fs.existsSync(logPath) && fs.statSync(logPath).size > 2 * 1024 * 1024) {
      fs.renameSync(logPath, `${logPath}.previous`);
    }
    fs.appendFileSync(logPath, clean);
  } catch { /* An unwritable diagnostic file must not crash the desktop. */ }
}

async function fail(error) {
  log(`${error.stack || error}\n`);
  if (failureShowing || quitting) return;
  failureShowing = true;
  ready = false;
  await dialog.showMessageBox(window && !window.isDestroyed() ? window : undefined, {
    type: 'error', title: 'ChemApp could not continue',
    message: 'The local ChemApp backend is unavailable.',
    detail: `${error.message}\n\nNo remote backend is used. Diagnostic log: ${logPath}\nCheck the Python runtime/dependencies and relaunch ChemApp.`,
    buttons: ['Quit'],
  });
  app.quit();
}

async function exportWorkspaceImage() {
  if (!ready || capturing || !window || window.isDestroyed()) return;
  capturing = true;
  try {
    await captureWorkspace({
      contents: window.webContents,
      choosePath: () => dialog.showSaveDialog(window, {
        title: '导出工作区截图 / Export workspace image',
        defaultPath: path.join(app.getPath('pictures'), `ChemApp-${new Date().toISOString().slice(0,10)}.png`),
        filters: [{name: 'PNG image', extensions: ['png']}],
      }),
      writeFile: (file, bytes) => fs.promises.writeFile(file, bytes),
    });
  } catch (error) {
    log(`Workspace image: ${error.message}\n`);
    if (!quitting) await dialog.showMessageBox(window, {type: 'error', message: '截图未保存 / Image was not saved', detail: error.message});
  } finally { capturing = false; }
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (window && !window.isDestroyed()) {
      if (window.isMinimized()) window.restore();
      window.focus();
    }
  });

  app.whenReady().then(async () => {
    const userData = app.getPath('userData');
    fs.mkdirSync(userData, { recursive: true });
    logPath = (!app.isPackaged && process.env.CHEMAPP_DESKTOP_LOG_PATH) || path.join(userData, 'desktop-backend.log');
    const privateSession = session.fromPartition(`chemapp-${randomUUID()}`, { cache: false });
    privateSession.setPermissionRequestHandler((_webContents, _permission, callback) => callback(false));
    privateSession.setPermissionCheckHandler(() => false);
    privateSession.setDevicePermissionHandler(() => false);
    // The backend never needs a proxy; ignoring inherited proxy settings prevents
    // accidentally sending loopback authentication headers to a proxy server.
    await privateSession.setProxy({ mode: 'direct' });
    backend = new Backend({
      paths: launchPaths({
        packaged: app.isPackaged, resourcesPath: process.resourcesPath,
        pythonOverride: app.isPackaged ? undefined : process.env.CHEMAPP_DESKTOP_PYTHON,
      }),
      dataDir: path.join(userData, 'data'), onLog: log,
    });
    const statePath = path.join(userData, 'window-state.json');
    let savedState = {};
    try { savedState = JSON.parse(fs.readFileSync(statePath, 'utf8')); } catch { /* first launch or invalid state */ }
    const bounds = initialWindowState(savedState, screen.getPrimaryDisplay().workArea);
    window = new BrowserWindow({
      x: bounds.x, y: bounds.y, width: bounds.width, height: bounds.height,
      minWidth: bounds.minWidth, minHeight: bounds.minHeight,
      title: 'ChemApp', backgroundColor: '#f7f8fa',
      webPreferences: {
        session: privateSession, contextIsolation: true, sandbox: true,
        nodeIntegration: false, webSecurity: true, allowRunningInsecureContent: false,
        webviewTag: false, devTools: !app.isPackaged,
      },
    });
    if (bounds.maximized) window.maximize();
    window.on('close', () => {
      try { fs.writeFileSync(statePath, JSON.stringify({ ...window.getNormalBounds(), maximized: window.isMaximized() })); } catch (error) { log(`Window preferences: ${error.message}\n`); }
    });
    window.webContents.on('will-prevent-unload', (event) => {
      const answer = dialog.showMessageBoxSync(window, {
        type: 'warning', title: 'ChemApp · 未保存的修改 / Unsaved changes',
        message: '继续编辑，还是放弃未保存的修改？',
        detail: 'This view has unsaved edits. Keep editing, or discard them and leave? Saved spectra and results will be kept.',
        buttons: ['继续编辑 / Keep editing', '放弃修改 / Discard changes'],
        defaultId: 0, cancelId: 0, noLink: true,
      });
      if (answer === 1) event.preventDefault();
    });
    Menu.setApplicationMenu(Menu.buildFromTemplate([
      { label: 'ChemApp', submenu: [{ label: '导出工作区截图 / Export workspace image', accelerator: 'CommandOrControl+Shift+S', click: () => { void exportWorkspaceImage(); } }, {type: 'separator'}, { role: 'quit' }] },
      { label: 'Edit', submenu: [{ role: 'undo' }, { role: 'redo' }, { type: 'separator' }, { role: 'cut' }, { role: 'copy' }, { role: 'paste' }, { role: 'selectAll' }] },
      { label: 'View', submenu: [{ role: 'reload' }, { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' }, { role: 'togglefullscreen' }] },
    ]));
    window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
    window.webContents.on('will-attach-webview', (event) => event.preventDefault());
    window.webContents.on('will-navigate', (event, url) => {
      if (!backend.origin || !allowedNavigation(url, backend.origin)) event.preventDefault();
    });
    window.webContents.on('will-redirect', (event, url) => {
      if (!backend.origin || !allowedNavigation(url, backend.origin)) event.preventDefault();
    });
    window.webContents.on('will-frame-navigate', (event) => {
      if (!event.isMainFrame || !backend.origin || !allowedNavigation(event.url, backend.origin)) event.preventDefault();
    });
    window.webContents.on('console-message', (details) => {
      if (details.level === 'warning' || details.level === 'error') log(`Renderer ${details.level}: ${details.message}\n`);
    });
    window.webContents.on('render-process-gone', (_event, details) => {
      if (!quitting) void fail(new Error(`Renderer stopped: ${details.reason}`));
    });
    const loadingURL = pathToFileURL(path.join(__dirname, 'loading.html')).href;
    privateSession.webRequest.onBeforeRequest((details, callback) => {
      const loading = allowedLoadingRequest(details, loadingURL, window.webContents.id);
      callback({ cancel: !loading && (!backend.origin || !allowedRequest(details, backend.origin, window.webContents.id)) });
    });
    privateSession.webRequest.onBeforeSendHeaders((details, callback) => {
      if (allowedLoadingRequest(details, loadingURL, window.webContents.id)) {
        callback({ requestHeaders: details.requestHeaders });
        return;
      }
      const headers = backend.origin && requestHeaders(details, backend.origin, window.webContents.id, backend.secrets);
      callback(headers ? { requestHeaders: headers } : { cancel: true });
    });
    privateSession.on('will-download', (event, item, contents) => {
      const url = item.getURL();
      if (contents !== window.webContents || !backend.origin || !(sameOrigin(url, backend.origin) || url.startsWith(`blob:${backend.origin}/`) || url.startsWith('data:image/'))) {
        event.preventDefault();
        return;
      }
      item.setSaveDialogOptions({ title: 'Save ChemApp export', defaultPath: path.join(app.getPath('downloads'), path.basename(item.getFilename())) });
    });
    // A static, script-free local loading screen; never displays launch secrets.
    await window.loadFile(path.join(__dirname, 'loading.html'));
    if (quitting || window.isDestroyed()) return;
    backend.on('unexpected-exit', (error) => { if (ready) void fail(error); });
    const origin = await backend.start();
    if (quitting || window.isDestroyed()) return;
    await window.loadURL(`${origin}/`);
    if (backend.exited) throw new Error('The local backend stopped while loading the interface.');
    ready = true;
  }).catch((error) => { void fail(error); });
}

app.on('window-all-closed', () => app.quit());
app.on('before-quit', (event) => {
  requestQuit({ event, window, alreadyQuitting: quitting, shutdown: () => {
    quitting = true;
    Promise.resolve(backend?.stop()).catch((error) => log(`Shutdown: ${error.message}\n`)).finally(() => app.quit());
  }});
});

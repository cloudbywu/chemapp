'use strict';
// Capture only this app's rendered content. The user chooses the local file;
// no screen-wide access, permissions, renderer IPC or external upload is used.
async function captureWorkspace({ contents, choosePath, writeFile }) {
  const snapshot = await contents.capturePage();
  const { canceled, filePath } = await choosePath();
  if (canceled || !filePath) return false;
  await writeFile(filePath, snapshot.toPNG());
  return true;
}
module.exports = { captureWorkspace };

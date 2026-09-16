export const MAX_ATTACHED_FILES = 5;
export const MAX_FILE_BYTES = 200 * 1024;
export const ACCEPTED_EXTENSIONS = [".txt", ".md", ".markdown", ".tex"];

let attachCounter = 0;

function cleanFileName(name) {
  return String(name || "dropped draft")
    .replace(/\.[a-z0-9]+$/i, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 140);
}

function titleFromText(text, fileName) {
  const lines = String(text || "").split(/\r?\n/);
  for (const raw of lines.slice(0, 20)) {
    const line = raw
      .replace(/^```.*$/, "")
      .replace(/^[#>*\-\d.)\s]+/, "")
      .trim();
    if (line.length >= 12) return line.slice(0, 140);
  }
  return cleanFileName(fileName) || "Dropped draft";
}

function extensionOf(name) {
  const match = String(name || "").toLowerCase().match(/\.[a-z0-9]+$/);
  return match ? match[0] : "";
}

export async function readAttachedFiles(fileList, existingCount = 0) {
  const files = Array.from(fileList || []);
  const attached = [];
  let skippedType = 0;
  let skippedSize = 0;
  for (const file of files) {
    if (existingCount + attached.length >= MAX_ATTACHED_FILES) break;
    if (!ACCEPTED_EXTENSIONS.includes(extensionOf(file.name))) {
      skippedType += 1;
      continue;
    }
    if (file.size > MAX_FILE_BYTES) {
      skippedSize += 1;
      continue;
    }
    try {
      const text = await file.text();
      const title = titleFromText(text, file.name);
      if (!title) continue;
      attachCounter += 1;
      attached.push({ id: `attach-${Date.now()}-${attachCounter}`, title, filename: file.name });
    } catch {
      skippedType += 1;
    }
  }
  return { attached, skippedType, skippedSize };
}

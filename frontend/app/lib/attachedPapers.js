export const MAX_ATTACHED_FILES = 5;
export const MAX_FILE_BYTES = 200 * 1024;
export const MAX_DOCUMENT_BYTES = 10 * 1024 * 1024;
export const MAX_CONTEXT_CHARS = 6000;
export const TEXT_EXTENSIONS = [".txt", ".md", ".markdown", ".tex"];
export const DOCUMENT_EXTENSIONS = [".pdf", ".docx"];
export const ACCEPTED_EXTENSIONS = [...TEXT_EXTENSIONS, ...DOCUMENT_EXTENSIONS];

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

function contextFromText(text) {
  return String(text || "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, MAX_CONTEXT_CHARS);
}

function extensionOf(name) {
  const match = String(name || "").toLowerCase().match(/\.[a-z0-9]+$/);
  return match ? match[0] : "";
}

function toBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || "").split(",", 2)[1] || "");
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

async function documentText(file) {
  const response = await fetch("/api/attachment-text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: file.name, data_base64: await toBase64(file) }),
  });
  if (!response.ok) throw new Error("Could not read document");
  const payload = await response.json();
  return String(payload.text || "");
}

export async function readAttachedFiles(fileList, existingCount = 0) {
  const files = Array.from(fileList || []);
  const attached = [];
  let skippedType = 0;
  let skippedSize = 0;
  for (const file of files) {
    if (existingCount + attached.length >= MAX_ATTACHED_FILES) break;
    const extension = extensionOf(file.name);
    const isDocument = DOCUMENT_EXTENSIONS.includes(extension);
    if (!ACCEPTED_EXTENSIONS.includes(extension)) {
      skippedType += 1;
      continue;
    }
    if (file.size > (isDocument ? MAX_DOCUMENT_BYTES : MAX_FILE_BYTES)) {
      skippedSize += 1;
      continue;
    }
    try {
      const text = isDocument ? await documentText(file) : await file.text();
      if (!text.trim()) {
        skippedType += 1;
        continue;
      }
      const title = titleFromText(text, file.name);
      if (!title) continue;
      attachCounter += 1;
      attached.push({
        id: `attach-${Date.now()}-${attachCounter}`,
        title,
        filename: file.name,
        context: contextFromText(text),
      });
    } catch {
      skippedType += 1;
    }
  }
  return { attached, skippedType, skippedSize };
}

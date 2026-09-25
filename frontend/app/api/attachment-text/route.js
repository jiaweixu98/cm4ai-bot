const BACKEND = (
  process.env.MATRIX_BACKEND_URL ||
  "http://127.0.0.1:8100"
).replace(/\/$/, "");

export async function POST(request) {
  try {
    const body = await request.json();
    const response = await fetch(`${BACKEND}/api/attachment-text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: String(body.filename || ""), data_base64: String(body.data_base64 || "") }),
      cache: "no-store",
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      return Response.json({ error: "Could not read this document." }, { status: response.status || 502 });
    }
    return Response.json({ text: String(payload.text || "") });
  } catch (error) {
    console.error("attachment-text proxy failed:", error);
    return Response.json({ error: "Could not read this document." }, { status: 502 });
  }
}

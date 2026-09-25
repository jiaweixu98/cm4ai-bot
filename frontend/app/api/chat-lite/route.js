const BACKEND = (
  process.env.MATRIX_BACKEND_URL ||
  "http://127.0.0.1:8100"
).replace(/\/$/, "");

export async function POST(request) {
  try {
    const body = await request.json();
    if (!String(body.user_input || "").trim()) {
      return Response.json({ error: "user_input is required" }, { status: 400 });
    }

    const response = await fetch(`${BACKEND}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: request.headers.get("accept") || "application/json",
      },
      body: JSON.stringify(body),
      signal: request.signal,
      cache: "no-store",
    });
    if (response.ok && response.body && (response.headers.get("content-type") || "").includes("text/event-stream")) {
      return new Response(response.body, {
        headers: {
          "Content-Type": "text/event-stream; charset=utf-8",
          "Cache-Control": "no-cache, no-transform",
          "X-Accel-Buffering": "no",
        },
      });
    }
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = response.status === 429
        ? String(payload.detail || "MATRIX is busy. Try again in a moment.")
        : "Chat is unavailable. Try again.";
      return Response.json({ error: detail }, { status: response.status || 502 });
    }
    return Response.json(payload);
  } catch (error) {
    if (error?.name === "AbortError") {
      return Response.json({ error: "Chat request was cancelled" }, { status: 499 });
    }
    console.error("chat-lite proxy failed:", error);
    return Response.json({ error: "Chat is unavailable. Try again." }, { status: 502 });
  }
}

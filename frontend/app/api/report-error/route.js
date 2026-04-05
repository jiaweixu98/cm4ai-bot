const REPORT_API_URL =
  (process.env.REPORT_API_URL || "").trim() ||
  (process.env.BRIDGE_REPORT_API_URL || "").trim() ||
  "http://127.0.0.1:5173/api/report-error";

export async function POST(request) {
  try {
    const body = await request.json();
    const feedback = String(body.feedback || "").trim();
    if (!feedback) {
      return Response.json({ error: "feedback is required" }, { status: 400 });
    }

    const response = await fetch(REPORT_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });
    const payload = await response.json().catch(() => ({ error: "Invalid response from report API" }));
    return Response.json(payload, { status: response.status });
  } catch (error) {
    console.error("Failed to proxy report error feedback (frontend route):", error);
    return Response.json({ error: "Failed to forward report error feedback" }, { status: 500 });
  }
}

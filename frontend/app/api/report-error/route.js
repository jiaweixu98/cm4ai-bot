import { mkdir, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";

const REPORT_OUTPUT_DIR = "/data/jiawei_data/bridge2aikg/work/data";

function sanitizeToken(value, fallback) {
  const cleaned = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
  return cleaned || fallback;
}

function createReportFilename(project, page) {
  const iso = new Date().toISOString().replace(/[:.]/g, "-");
  const random = Math.random().toString(36).slice(2, 8);
  return `report_error_${project}_${page}_${iso}_${random}.json`;
}

function resolveReportFolder(folder) {
  const normalized = sanitizeToken(folder, "matrix_error");
  const allowed = new Set(["matrix_error", "kg_error", "general_feedback"]);
  return allowed.has(normalized) ? normalized : "matrix_error";
}

export async function POST(request) {
  try {
    const body = await request.json();
    const feedback = String(body.feedback || "").trim();
    if (!feedback) {
      return Response.json({ error: "feedback is required" }, { status: 400 });
    }

    const project = sanitizeToken(body.project || "cm4ai-bot", "cm4ai-bot");
    const page = sanitizeToken(body.page || "unknown-page", "unknown-page");
    const reportFolder = resolveReportFolder(body.report_folder || "matrix_error");
    const filename = createReportFilename(project, page);
    const outputPath = join(REPORT_OUTPUT_DIR, reportFolder, basename(filename));

    const payload = {
      report_type: "error_feedback",
      submitted_at: new Date().toISOString(),
      project,
      page,
      report_folder: reportFolder,
      feedback,
      context: body.context || {},
      current_url: body.current_url || null,
      user_agent: body.user_agent || null,
      source: "cm4ai-bot-frontend-route",
    };

    await mkdir(join(REPORT_OUTPUT_DIR, reportFolder), { recursive: true });
    await writeFile(outputPath, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");

    return Response.json({
      ok: true,
      filename,
      path: outputPath,
    });
  } catch (error) {
    console.error("Failed to save report error feedback (frontend route):", error);
    return Response.json({ error: "Failed to save report error feedback" }, { status: 500 });
  }
}

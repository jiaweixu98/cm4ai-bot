export const BRIDGE_MSG = {
  OPEN_PERSON: "bridge2ai-open-person",
  SAVE_PERSON: "bridge2ai-save-person",
  UNSAVE_PERSON: "bridge2ai-unsave-person",
  REQUEST_SAVED: "bridge2ai-request-saved",
  SAVED_PEOPLE: "bridge2ai-saved-people",
  OPEN_PERSON_RESULT: "bridge2ai-open-person-result",
};

export function inBridgeIframe() {
  try {
    return typeof window !== "undefined" && window.parent && window.parent !== window;
  } catch {
    return false;
  }
}

export function postToBridge(payload) {
  if (!inBridgeIframe()) return;
  window.parent.postMessage(payload, "*");
}

export function isBridgeOrigin(origin) {
  if (!origin || typeof window === "undefined") return false;
  if (origin === window.location.origin) return true;
  const allowed = new Set([
    "http://127.0.0.1:4173",
    "http://localhost:4173",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
  ]);
  if (allowed.has(origin)) return true;
  try {
    const referrer = document.referrer ? new URL(document.referrer).origin : "";
    return Boolean(referrer) && origin === referrer;
  } catch {
    return false;
  }
}

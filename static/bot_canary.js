/*
 * Bot Engine — Canary endpoints (P1 #8)
 * =======================================
 * This file is deliberately shipped to the browser but its code paths are
 * never executed at runtime. Only bots that parse JS bundles looking for
 * API endpoints to enumerate will follow the URLs embedded here — and
 * requesting any of them scores the client at 100 (bad) immediately.
 *
 * Serve this file from a path referenced in your main bundle via a
 * dead-code path, e.g.:
 *
 *     // At the bottom of chatbot.js
 *     if (window.__BOT_CANARY__) {
 *       // Never true in a real browser run
 *       import('./bot_canary.js').then(m => m.fetchInternalModels());
 *     }
 *
 * The `__BOT_CANARY__` global is never set, so the module never loads in
 * real clients. Bots that ignore control-flow and just scrape string
 * literals out of the bundle will see the URLs below and probe them.
 *
 * Keep these endpoints in sync with CONFIG["CANARY_PATHS"] in the bot
 * engine so that hits are actually scored.
 */

// Fake API version that doesn't exist — bot-bait only.
const CANARY_BASE = "/api/v2/_internal";

export async function fetchInternalModels() {
  // Dead code. Real users never reach this function.
  return fetch(`${CANARY_BASE}/models_internal`, {
    headers: { "X-Internal-Call": "1" },
  }).then(r => r.json());
}

export async function fetchAdminKeys() {
  // Dead code. Real users never reach this function.
  return fetch(`${CANARY_BASE}/admin/api_keys`, {
    method: "GET",
  }).then(r => r.json());
}

export async function exportTrainingData() {
  // Dead code. Real users never reach this function.
  return fetch(`${CANARY_BASE}/training/export.jsonl`, {
    method: "POST",
    body: JSON.stringify({ format: "jsonl", include_users: true }),
  });
}

// Canary URL strings harvested by JS-scrapers even without executing code.
// Keep these literal so regex-based scrapers will catch them.
export const CANARY_URLS = [
  "/api/v2/_internal/models_internal",
  "/api/v2/_internal/admin/api_keys",
  "/api/v2/_internal/training/export.jsonl",
  "/api/v2/_internal/users/dump",
];

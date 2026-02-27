// /js/api.js
/**
 * API helpers for:
 * - Production (SWA frontend + App Service backend): call backend via full base URL
 * - Local dev: override via DevTools: window.LOCAL_API_BASE="http://127.0.0.1:7071"
 *
 * Usage:
 *   fetch(apiUrl("/latest"))
 *   const data = await apiGetJson("/latest")
 */

(function () {
  // Optional flag: when true, disable all backend calls (static demo mode)
  window.STATIC_ONLY = window.STATIC_ONLY ?? false;

  // ✅ Default backend base (production)
  window.DEFAULT_API_BASE =
    window.DEFAULT_API_BASE ||
    "https://rgnfunction-ecdqdxfffqcmb0e5.ukwest-01.azurewebsites.net";

  // ✅ Local override for dev:
  //   window.LOCAL_API_BASE = "http://127.0.0.1:7071";
  // Set in DevTools or before api.js loads.
  window.LOCAL_API_BASE = window.LOCAL_API_BASE || "";

  // Expose prefix for reuse / debugging
  window.API_PREFIX = window.API_PREFIX || "/api";

  /**
   * Build an API URL.
   * - apiUrl("/latest") => "<BASE>/api/latest"
   * - apiUrl("/api/latest") keeps prefix
   * - apiUrl("https://...") returns as-is
   * - If STATIC_ONLY=true => return null
   */
  window.apiUrl = function apiUrl(path) {
    if (window.STATIC_ONLY) return null;

    if (typeof path !== "string" || !path.length) path = "/";
    // allow full URL passthrough
    if (/^https?:\/\//i.test(path)) return path;

    if (!path.startsWith("/")) path = "/" + path;

    // If caller already passed "/api/..", keep it; otherwise prepend "/api"
    const p = path.startsWith(window.API_PREFIX + "/")
      ? path
      : window.API_PREFIX + path;

    // Choose base:
    // 1) LOCAL_API_BASE if provided
    // 2) DEFAULT_API_BASE otherwise
    const base = (window.LOCAL_API_BASE || window.DEFAULT_API_BASE || "").replace(
      /\/$/,
      ""
    );

    // If no base (rare), fall back to same-origin
    return base ? base + p : p;
  };

  /**
   * GET JSON helper (consistent error handling)
   */
  window.apiGetJson = async function apiGetJson(path, options = {}) {
    const url = window.apiUrl(path);
    if (!url) throw new Error("STATIC_ONLY enabled: apiGetJson blocked");

    const res = await fetch(url, {
      method: "GET",
      headers: { Accept: "application/json", ...(options.headers || {}) },
      ...options,
    });

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`GET ${path} failed: ${res.status} ${text}`.trim());
    }
    return res.json();
  };

  /**
   * POST JSON helper
   */
  window.apiPostJson = async function apiPostJson(path, body, options = {}) {
    const url = window.apiUrl(path);
    if (!url) throw new Error("STATIC_ONLY enabled: apiPostJson blocked");

    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        ...(options.headers || {}),
      },
      body: JSON.stringify(body ?? {}),
      ...options,
    });

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`POST ${path} failed: ${res.status} ${text}`.trim());
    }
    return res.json();
  };

  // Optional: quick console check
  // console.log("api.js loaded:", {
  //   STATIC_ONLY: window.STATIC_ONLY,
  //   LOCAL_API_BASE: window.LOCAL_API_BASE,
  //   DEFAULT_API_BASE: window.DEFAULT_API_BASE,
  //   API_PREFIX: window.API_PREFIX,
  // });
})();
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
  // ✅ Default to App Service backend in production (Route B)
  // You can override this in DevTools for local testing:
  //   window.LOCAL_API_BASE = "http://127.0.0.1:7071";
  // Or set earlier in HTML before loading api.js.
  window.LOCAL_API_BASE =
    window.LOCAL_API_BASE ||
    "https://rgnfunction-ecdqdxfffqcmb0e5.ukwest-01.azurewebsites.net";

  // Expose prefix for reuse / debugging
  window.API_PREFIX = window.API_PREFIX || "/api";

  /**
   * Build an API URL.
   * - apiUrl("/latest") => "<LOCAL_API_BASE>/api/latest"
   * - apiUrl("/api/latest") keeps as-is
   */
  window.apiUrl = function apiUrl(path) {
    if (typeof path !== "string" || !path.length) path = "/";
    if (!path.startsWith("/")) path = "/" + path;

    // If caller already passed "/api/..", keep it; otherwise prepend "/api"
    const p = path.startsWith(window.API_PREFIX + "/") ? path : window.API_PREFIX + path;

    // Route B: always use full backend base (App Service) by default
    if (window.LOCAL_API_BASE) {
      return window.LOCAL_API_BASE.replace(/\/$/, "") + p;
    }

    // Fallback (should rarely be used): same-origin
    return p;
  };

  /**
   * GET JSON helper (consistent error handling)
   */
  window.apiGetJson = async function apiGetJson(path, options = {}) {
    const res = await fetch(window.apiUrl(path), {
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
    const res = await fetch(window.apiUrl(path), {
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
  // console.log("api.js loaded:", { LOCAL_API_BASE: window.LOCAL_API_BASE, API_PREFIX: window.API_PREFIX });
})();
// /js/api.js
/**
 * API helpers for:
 * - Azure Static Web Apps (production): call backend via "/api/..."
 * - Local dev (Azure Functions Core Tools): set LOCAL_API_BASE="http://127.0.0.1:7071"
 *
 * Usage:
 *   fetch(apiUrl("/latest"))
 *   const data = await apiGetJson("/rule_data")
 */

(function () {
    // Allow override from:
    // - DevTools console: window.LOCAL_API_BASE="http://127.0.0.1:7071"
    // - Or set earlier in HTML before loading api.js
    window.LOCAL_API_BASE = window.LOCAL_API_BASE || "";
  
    // Expose prefix for reuse / debugging
    window.API_PREFIX = window.API_PREFIX || "/api";
  
    /**
     * Build an API URL.
     * - apiUrl("/latest") => "/api/latest" (production on SWA)
     * - with LOCAL_API_BASE => "http://127.0.0.1:7071/api/latest"
     */
    window.apiUrl = function apiUrl(path) {
      if (typeof path !== "string" || !path.length) path = "/";
      if (!path.startsWith("/")) path = "/" + path;
  
      // If caller already passed "/api/..", keep it; otherwise prepend "/api"
      const p = path.startsWith(window.API_PREFIX + "/")
        ? path
        : window.API_PREFIX + path;
  
      if (window.LOCAL_API_BASE) {
        return window.LOCAL_API_BASE.replace(/\/$/, "") + p;
      }
      return p; // SWA production
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
// Thin wrapper around the FastAPI REST endpoints.
async function req(url, opts) {
  let r;
  try { r = await fetch(url, opts); } catch { throw new Error("Cannot reach backend. Is it running on port 8000?"); }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Request failed");
  return data;
}
export const api = {
  status: () => req("/api/status"),
  info: () => req("/api/dataset/info"),
  images: (offset = 0, limit = 40) => req(`/api/images?offset=${offset}&limit=${limit}`),
  sample: (name) => req(`/api/analyze/sample?name=${encodeURIComponent(name)}`),
  upload: (file) => { const f = new FormData(); f.append("file", file); return req("/api/analyze/upload", { method: "POST", body: f }); },
  dsp: (p) => req(`/api/dsp?${new URLSearchParams(p)}`),
  health: (greenness_level, readings) => req("/api/health", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ greenness_level, readings }) }),
};

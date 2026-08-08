/** Display helpers. Presentation only - no prediction logic lives here. */

/** 0.971 -> "97.1%" */
export function toPercent(value, digits = 1) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '--'
  return `${(value * 100).toFixed(digits)}%`
}

/** "HUMAN_AGENT" -> "HUMAN AGENT" */
export function humanise(value) {
  if (!value) return ''
  return String(value).replace(/_/g, ' ')
}

/** "HIGH" -> "HIGH PRIORITY" */
export function priorityHeading(label) {
  return label ? `${label} PRIORITY` : 'PRIORITY'
}

/** Lower-cased label for CSS class lookups, with a safe fallback. */
export function classKey(value, allowed, fallback) {
  const key = String(value || '').toLowerCase()
  return allowed.includes(key) ? key : fallback
}

/** Session ids: crypto.randomUUID needs a secure context, so keep a fallback. */
export function newSessionId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID()
  return `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

export function shortId(sessionId, length = 8) {
  return String(sessionId || '').replace(/-/g, '').slice(0, length)
}

/** 560078007 -> "560M" */
export function formatParameters(count) {
  if (typeof count !== 'number' || !Number.isFinite(count)) return null
  if (count >= 1e9) return `${(count / 1e9).toFixed(1)}B`
  if (count >= 1e6) return `${Math.round(count / 1e6)}M`
  return String(count)
}

export function formatTime(date = new Date()) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

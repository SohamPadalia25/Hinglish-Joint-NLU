/**
 * The only place the frontend talks to the backend.
 *
 * The backend is the single source of truth for intent, sentiment, slots,
 * priority, context and the reply text. Nothing here computes or guesses any of
 * those - it forwards what the API returned.
 */

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
).replace(/\/+$/, '')

const DEFAULT_TIMEOUT_MS = 30000

/** Error carrying a message that is safe to show a user. */
export class ApiError extends Error {
  constructor(message, { status = null, kind = 'error' } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.kind = kind
  }
}

async function request(path, { method = 'GET', body, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)

  let response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    })
  } catch (error) {
    // fetch rejects for aborts and for anything that never reached the server.
    if (error.name === 'AbortError') {
      throw new ApiError(
        'The request timed out. The model runs on CPU, so the first reply can be slow - please try again.',
        { kind: 'timeout' },
      )
    }
    throw new ApiError(
      `Cannot reach the backend at ${API_BASE_URL}. Make sure it is running, then try again.`,
      { kind: 'offline' },
    )
  } finally {
    clearTimeout(timer)
  }

  if (!response.ok) {
    // Surface the API's own detail when it is a readable string; never leak a
    // Python traceback into the UI.
    let detail = ''
    try {
      const payload = await response.json()
      if (typeof payload?.detail === 'string') detail = payload.detail
    } catch {
      /* body was not JSON - fall through to the generic message */
    }

    if (response.status === 422) {
      throw new ApiError(detail || 'That message could not be processed. Please rephrase it.', {
        status: 422,
        kind: 'validation',
      })
    }
    if (response.status === 404) {
      throw new ApiError(detail || 'That conversation no longer exists. Start a new one.', {
        status: 404,
        kind: 'notfound',
      })
    }
    throw new ApiError(
      `The backend returned an error (${response.status}). Please try again.`,
      { status: response.status, kind: 'server' },
    )
  }

  try {
    return await response.json()
  } catch {
    throw new ApiError('The backend sent a response the app could not read.', {
      kind: 'malformed',
    })
  }
}

/** Shape check, so a malformed payload fails here rather than deep in a component. */
function assertChatResult(data) {
  const ok =
    data &&
    typeof data === 'object' &&
    data.nlu &&
    typeof data.nlu.intent === 'string' &&
    data.priority &&
    typeof data.priority.label === 'string' &&
    typeof data.response === 'string'

  if (!ok) {
    throw new ApiError('The backend sent an unexpected response format.', { kind: 'malformed' })
  }
  return data
}

/**
 * Send one conversational turn.
 * @param {string} sessionId reused across the whole conversation
 * @param {string} message raw user text
 */
export async function sendMessage(sessionId, message) {
  const text = (message ?? '').trim()
  if (!text) {
    throw new ApiError('Type a message before sending.', { kind: 'validation' })
  }
  const data = await request('/api/chat', {
    method: 'POST',
    body: { session_id: sessionId, message: text },
  })
  return assertChatResult(data)
}

/** Liveness + what the backend actually has loaded. */
export async function healthCheck() {
  return request('/health', { timeoutMs: 5000 })
}

/** Forget a conversation server-side. Best-effort: a failure is not user-facing. */
export async function clearSession(sessionId) {
  try {
    return await request(`/api/session/${encodeURIComponent(sessionId)}`, {
      method: 'DELETE',
      timeoutMs: 5000,
    })
  } catch {
    return null
  }
}

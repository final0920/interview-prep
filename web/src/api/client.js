// HTTP + WebSocket client wired to the section-10 API contract.
// All responses are shaped { ok: bool, data, message }.

const BASE = '/api'

async function request(method, path, body) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  }
  if (body !== undefined) opts.body = JSON.stringify(body)
  const res = await fetch(BASE + path, opts)
  const json = await res.json()
  if (!json.ok) throw new Error(json.message || 'request failed')
  return json.data
}

// POST /api/interview/start { target_role } -> { session_id, question }
export function startInterview(targetRole) {
  return request('POST', '/interview/start', { target_role: targetRole })
}

// POST /api/interview/answer { session_id, answer } -> { evaluation, next }
export function submitAnswer(sessionId, answer) {
  return request('POST', '/interview/answer', { session_id: sessionId, answer })
}

// GET /api/interview/{session_id} -> InterviewState
export function getSession(sessionId) {
  return request('GET', `/interview/${sessionId}`)
}

// GET /api/memory -> { episodes, semantic, weakpoints }
export function getMemory() {
  return request('GET', '/memory')
}

// GET /api/resume -> { profile, health_report }
export function getResume() {
  return request('GET', '/resume')
}

// GET /api/review -> { schedule, quality_report }
export function getReview() {
  return request('GET', '/review')
}

// WS /interview/ws
// Returns a WebSocket. Caller attaches onmessage / onclose.
// Messages from server: { type: "question"|"followup"|"score"|"done", payload }
export function openInterviewWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  return new WebSocket(`${proto}://${location.host}/interview/ws`)
}

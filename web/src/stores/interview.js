import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { startInterview, submitAnswer, openInterviewWS } from '../api/client'

// A single turn in the conversation log.
// type: 'question' | 'followup' | 'answer' | 'score' | 'done'
function makeTurn(type, payload) {
  return { id: Date.now() + Math.random(), type, payload, ts: Date.now() }
}

export const useInterviewStore = defineStore('interview', () => {
  const sessionId = ref(null)
  const targetRole = ref('AI Application / LLM Engineering')
  const turns = ref([])           // conversation log
  const currentScore = ref(null)  // latest AnswerEvaluation
  const status = ref('idle')      // idle | active | done | error
  const errorMsg = ref('')
  const wsRef = ref(null)         // active WebSocket

  const isActive = computed(() => status.value === 'active')
  const isDone = computed(() => status.value === 'done')

  function resetSession() {
    turns.value = []
    currentScore.value = null
    sessionId.value = null
    status.value = 'idle'
    errorMsg.value = ''
    if (wsRef.value) {
      wsRef.value.close()
      wsRef.value = null
    }
  }

  // REST-based flow (start + submit answers)
  async function start(role) {
    resetSession()
    if (role) targetRole.value = role
    status.value = 'active'
    try {
      const data = await startInterview(targetRole.value)
      sessionId.value = data.session_id
      turns.value.push(makeTurn('question', data.question))
    } catch (e) {
      status.value = 'error'
      errorMsg.value = e.message
    }
  }

  async function answer(text) {
    if (!sessionId.value) return
    turns.value.push(makeTurn('answer', text))
    try {
      const data = await submitAnswer(sessionId.value, text)
      if (data.evaluation) {
        currentScore.value = data.evaluation
        turns.value.push(makeTurn('score', data.evaluation))
      }
      if (data.next) {
        if (data.next === 'done') {
          turns.value.push(makeTurn('done', null))
          status.value = 'done'
        } else if (data.next.type === 'followup') {
          turns.value.push(makeTurn('followup', data.next))
        } else {
          turns.value.push(makeTurn('question', data.next))
        }
      }
    } catch (e) {
      status.value = 'error'
      errorMsg.value = e.message
    }
  }

  // WebSocket-based streaming flow
  function connectWS() {
    if (wsRef.value) wsRef.value.close()
    const ws = openInterviewWS()
    wsRef.value = ws

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data)
        // msg: { type: question|followup|score|done, payload }
        if (msg.type === 'score') currentScore.value = msg.payload
        if (msg.type === 'done') status.value = 'done'
        turns.value.push(makeTurn(msg.type, msg.payload))
      } catch {
        // ignore parse errors
      }
    }
    ws.onerror = () => {
      status.value = 'error'
      errorMsg.value = 'WebSocket error'
    }
    ws.onclose = () => {
      wsRef.value = null
    }
  }

  function sendWSAnswer(text) {
    if (!wsRef.value || wsRef.value.readyState !== WebSocket.OPEN) return
    turns.value.push(makeTurn('answer', text))
    wsRef.value.send(JSON.stringify({ type: 'answer', payload: text }))
  }

  return {
    sessionId, targetRole, turns, currentScore, status, errorMsg,
    isActive, isDone,
    resetSession, start, answer, connectWS, sendWSAnswer,
  }
})

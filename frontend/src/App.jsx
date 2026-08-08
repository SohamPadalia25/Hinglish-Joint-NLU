import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import AnalysisPanel from './components/AnalysisPanel.jsx'
import ChatInput from './components/ChatInput.jsx'
import ChatPanel from './components/ChatPanel.jsx'
import ErrorBanner from './components/ErrorBanner.jsx'
import ExampleQueries from './components/ExampleQueries.jsx'
import Header from './components/Header.jsx'
import ModelInfo from './components/ModelInfo.jsx'
import { clearSession, healthCheck, sendMessage } from './services/api.js'
import { formatTime, newSessionId } from './utils/format.js'
import styles from './App.module.css'

// Turn 1 establishes the order, turn 2 is the short follow-up that can only be
// understood against it. Both go through the same session id.
const CONTEXT_DEMO = ['Maine order #45231 cancel kiya tha', 'Refund kab milega?']

let messageCounter = 0
const nextId = () => `m${++messageCounter}`

export default function App() {
  const [sessionId, setSessionId] = useState(newSessionId)
  const [messages, setMessages] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [health, setHealth] = useState({ status: 'checking' })

  // `loading` is captured per render, so a sequenced demo would read a stale
  // value. A ref gives the guard the current state.
  const busyRef = useRef(false)

  const checkHealth = useCallback(async () => {
    try {
      setHealth(await healthCheck())
    } catch {
      setHealth({ status: 'offline' })
    }
  }, [])

  useEffect(() => {
    checkHealth()
  }, [checkHealth])

  const send = useCallback(
    async (text) => {
      const trimmed = (text ?? '').trim()
      if (!trimmed) {
        setError('Type a message before sending.')
        return false
      }
      if (busyRef.current) return false

      busyRef.current = true
      setLoading(true)
      setError(null)
      setMessages((current) => [
        ...current,
        { id: nextId(), role: 'user', text: trimmed, time: formatTime() },
      ])

      try {
        // Everything shown in the UI comes from this response. Nothing is
        // predicted, derived or reworded on the client.
        const result = await sendMessage(sessionId, trimmed)
        const botId = nextId()
        setMessages((current) => [
          ...current,
          { id: botId, role: 'bot', text: result.response, time: formatTime(), result },
        ])
        setSelectedId(botId)
        return true
      } catch (apiError) {
        setError(apiError.message || 'Something went wrong. Please try again.')
        if (apiError.kind === 'offline') checkHealth()
        return false
      } finally {
        busyRef.current = false
        setLoading(false)
      }
    },
    [sessionId, checkHealth],
  )

  const runContextDemo = useCallback(async () => {
    const firstTurnSucceeded = await send(CONTEXT_DEMO[0])
    if (firstTurnSucceeded) await send(CONTEXT_DEMO[1])
  }, [send])

  const startNewConversation = useCallback(() => {
    if (busyRef.current) return
    // The backend only materialises a session on its first message, so asking it
    // to delete an untouched one is a guaranteed 404. Skip it.
    if (messages.length > 0) clearSession(sessionId)
    setSessionId(newSessionId())
    setMessages([])
    setSelectedId(null)
    setError(null)
  }, [sessionId, messages.length])

  const selectedResult = useMemo(() => {
    const selected = messages.find((message) => message.id === selectedId)
    if (selected?.result) return selected.result
    // Fall back to the newest analysed turn.
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].result) return messages[i].result
    }
    return null
  }, [messages, selectedId])

  return (
    <div className={styles.app}>
      <Header
        health={health}
        sessionId={sessionId}
        turnCount={messages.filter((message) => message.role === 'user').length}
        onNewConversation={startNewConversation}
        disabled={loading}
      />

      <main className={styles.layout}>
        <section className={styles.chatColumn}>
          {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

          <ChatPanel
            messages={messages}
            loading={loading}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />

          <ChatInput onSend={send} disabled={loading} />

          <ExampleQueries
            onPick={send}
            onContextDemo={runContextDemo}
            disabled={loading}
          />
        </section>

        <aside className={styles.analysisColumn}>
          <AnalysisPanel result={selectedResult} />
          <ModelInfo health={health} />
        </aside>
      </main>
    </div>
  )
}

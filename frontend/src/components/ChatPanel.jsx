import { useEffect, useRef } from 'react'

import MessageBubble from './MessageBubble.jsx'
import styles from './ChatPanel.module.css'

export default function ChatPanel({ messages, loading, selectedId, onSelect }) {
  const endRef = useRef(null)

  // Keep the newest turn (or the typing indicator) in view.
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages.length, loading])

  return (
    <div className={`card ${styles.panel}`}>
      <div className={styles.head}>
        <h2 className={styles.heading}>Conversation</h2>
        <span className={styles.hint}>Select a reply to view its analysis</span>
      </div>

      <div className={styles.scroll} role="log" aria-live="polite" aria-label="Conversation">
        {messages.length === 0 && !loading && (
          <div className={styles.empty}>
            <p className={styles.emptyTitle}>No messages yet</p>
            <p className={styles.emptyBody}>
              Type a Hinglish support query below, or pick one of the examples to start.
            </p>
          </div>
        )}

        {messages.map((message) => (
          <MessageBubble
            key={message.id}
            message={message}
            selected={message.id === selectedId}
            onSelect={onSelect}
          />
        ))}

        {loading && (
          <div className={styles.typing}>
            <span className={styles.typingDots} aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
            Analysing with XLM-RoBERTa Large...
          </div>
        )}

        <div ref={endRef} />
      </div>
    </div>
  )
}

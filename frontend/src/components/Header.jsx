import { shortId } from '../utils/format.js'
import styles from './Header.module.css'

const STATUS_TEXT = {
  ok: 'Backend connected',
  checking: 'Connecting...',
  loading: 'Model loading...',
  offline: 'Backend offline',
}

export default function Header({ health, sessionId, turnCount, onNewConversation, disabled }) {
  const status = health?.status === 'ok' && health?.model_loaded ? 'ok' : health?.status || 'offline'
  const statusClass = styles[status] || styles.offline

  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <div className={styles.titles}>
          <h1 className={styles.title}>Hinglish AI Customer Support</h1>
          <p className={styles.subtitle}>Context-Aware Code-Switched NLU</p>
        </div>

        <div className={styles.meta}>
          <span className={`${styles.status} ${statusClass}`}>
            <span className={styles.dot} aria-hidden="true" />
            {STATUS_TEXT[status] || 'Backend offline'}
          </span>

          <span className={styles.session} title={sessionId}>
            Session <code>{shortId(sessionId)}</code>
            <span className={styles.turns}>
              {turnCount} {turnCount === 1 ? 'turn' : 'turns'}
            </span>
          </span>

          <button
            type="button"
            className={styles.newButton}
            onClick={onNewConversation}
            disabled={disabled}
          >
            New Conversation
          </button>
        </div>
      </div>
    </header>
  )
}

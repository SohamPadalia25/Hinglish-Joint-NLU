import { classKey } from '../utils/format.js'
import styles from './MessageBubble.module.css'

const PRIORITIES = ['high', 'medium', 'low']

export default function MessageBubble({ message, selected, onSelect }) {
  const isUser = message.role === 'user'
  const result = message.result
  const analysable = Boolean(result) && typeof onSelect === 'function'

  const priority = result?.priority?.label
  const priorityClass = styles[classKey(priority, PRIORITIES, 'medium')]

  const body = (
    <>
      <div className={styles.role}>
        {isUser ? 'You' : 'Support Assistant'}
        <span className={styles.time}>{message.time}</span>
      </div>

      {/* Rendered as plain text exactly as the backend returned it. */}
      <p className={styles.text}>{message.text}</p>

      {result && (
        <div className={styles.tags}>
          <span className={styles.tag}>{result.nlu.intent}</span>
          <span className={`${styles.tag} ${priorityClass}`}>{priority}</span>
          {result.context?.used && <span className={styles.contextTag}>Context used</span>}
        </div>
      )}
    </>
  )

  const className = [
    styles.row,
    isUser ? styles.user : styles.bot,
    selected ? styles.selected : '',
  ]
    .filter(Boolean)
    .join(' ')

  if (!analysable) {
    return (
      <div className={className}>
        <div className={styles.bubble}>{body}</div>
      </div>
    )
  }

  return (
    <div className={className}>
      <button
        type="button"
        className={`${styles.bubble} ${styles.clickable}`}
        onClick={() => onSelect(message.id)}
        aria-pressed={selected}
        aria-label="Show the analysis for this reply"
      >
        {body}
      </button>
    </div>
  )
}

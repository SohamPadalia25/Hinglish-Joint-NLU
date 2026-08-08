import { classKey, humanise, priorityHeading, toPercent } from '../utils/format.js'
import styles from './PriorityCard.module.css'

const LEVELS = ['high', 'medium', 'low']

/**
 * Priority and routing, straight from the backend's rule engine.
 *
 * Routing is displayed as returned rather than derived from the priority label:
 * they usually agree (HIGH -> HUMAN_AGENT), but a callback request is MEDIUM and
 * still routes to a human, and showing the real value keeps that honest.
 */
export default function PriorityCard({ priority }) {
  if (!priority) return null

  const level = classKey(priority.label, LEVELS, 'medium')

  return (
    <div className={`${styles.card} ${styles[level]}`}>
      <div className={styles.main}>
        <span className={styles.heading}>{priorityHeading(priority.label)}</span>
        <span className={styles.routing}>{humanise(priority.routing)}</span>
      </div>

      <div className={styles.scoreRow}>
        <span className={styles.scoreLabel}>Priority score</span>
        <span className={styles.scoreValue}>{toPercent(priority.score)}</span>
      </div>

      <div
        className={styles.track}
        role="progressbar"
        aria-valuenow={Math.round((priority.score || 0) * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Priority score"
      >
        <span
          className={styles.fill}
          style={{ '--fill': `${Math.min(Math.max(priority.score || 0, 0), 1) * 100}%` }}
        />
      </div>
    </div>
  )
}

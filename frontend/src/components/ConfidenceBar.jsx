import { toPercent } from '../utils/format.js'
import styles from './ConfidenceBar.module.css'

/**
 * A labelled 0-1 confidence read-out.
 *
 * The colour comes from `tone` (a CSS class); only the bar width is dynamic,
 * passed as a custom property rather than a hardcoded inline style rule.
 */
export default function ConfidenceBar({ value, tone = 'accent', label = 'Confidence' }) {
  const safe = typeof value === 'number' && Number.isFinite(value) ? Math.min(Math.max(value, 0), 1) : 0

  return (
    <div className={styles.wrapper}>
      <div className={styles.row}>
        <span className={styles.label}>{label}</span>
        <span className={styles.value}>{toPercent(value)}</span>
      </div>
      <div
        className={styles.track}
        role="progressbar"
        aria-valuenow={Math.round(safe * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label}
      >
        <span
          className={`${styles.fill} ${styles[tone] || styles.accent}`}
          style={{ '--fill': `${safe * 100}%` }}
        />
      </div>
    </div>
  )
}

import { toPercent } from '../utils/format.js'
import styles from './EntityList.module.css'

/**
 * Extracted slot spans.
 *
 * The backend already decodes BIO tags into typed spans, so `slot.type` is
 * "ORDER_ID", never "B-ORDER_ID". Nothing here parses tags.
 */
export default function EntityList({ slots = [], inherited = [] }) {
  const hasAny = slots.length > 0 || inherited.length > 0

  if (!hasAny) {
    return <p className={styles.empty}>No entities detected in this message.</p>
  }

  return (
    <ul className={styles.list}>
      {slots.map((slot, index) => (
        <li key={`own-${slot.type}-${index}`} className={styles.item}>
          <div className={styles.top}>
            <span className={styles.type}>{slot.type}</span>
            {typeof slot.confidence === 'number' && (
              <span className={styles.confidence}>{toPercent(slot.confidence)}</span>
            )}
          </div>
          <span className={styles.value}>{slot.value}</span>
        </li>
      ))}

      {inherited.map((slot, index) => (
        <li key={`ctx-${slot.type}-${index}`} className={`${styles.item} ${styles.inherited}`}>
          <div className={styles.top}>
            <span className={styles.type}>{slot.type}</span>
            <span className={styles.source}>from context</span>
          </div>
          <span className={styles.value}>{slot.value}</span>
        </li>
      ))}
    </ul>
  )
}

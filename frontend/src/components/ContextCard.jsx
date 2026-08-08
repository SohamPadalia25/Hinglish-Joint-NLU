import { useState } from 'react'

import styles from './ContextCard.module.css'

/**
 * Whether this turn was interpreted against earlier ones.
 *
 * The backend's own `reason` string names its internal signals
 * ("short_query", "follow_up_marker", ...), which is noise during a demo. The
 * card leads with a plain-language summary and keeps the exact backend reason
 * one click away, so nothing is hidden.
 */
export default function ContextCard({ context }) {
  const [showDetail, setShowDetail] = useState(false)

  const used = Boolean(context?.used)
  const previousIntent = context?.previous_intent
  const inherited = context?.inherited_slots || []

  return (
    <div className={`${styles.card} ${used ? styles.used : styles.unused}`}>
      <div className={styles.head}>
        <span className={styles.label}>Context Used</span>
        <span className={`${styles.badge} ${used ? styles.badgeYes : styles.badgeNo}`}>
          {used ? 'YES' : 'NO'}
        </span>
      </div>

      <p className={styles.summary}>
        {used
          ? 'Current query interpreted using the previous conversation.'
          : 'Query is self-contained; earlier turns were not needed.'}
      </p>

      {used && (previousIntent || inherited.length > 0) && (
        <div className={styles.previous}>
          <span className={styles.previousLabel}>Previous context</span>
          <div className={styles.chips}>
            {inherited.map((slot, index) => (
              <span key={`${slot.type}-${index}`} className={styles.chip}>
                {slot.type === 'ORDER_ID' ? 'Order' : slot.type} {slot.value}
              </span>
            ))}
            {previousIntent && (
              <span className={`${styles.chip} ${styles.intentChip}`}>{previousIntent}</span>
            )}
          </div>
        </div>
      )}

      {context?.reason && (
        <>
          <button
            type="button"
            className={styles.toggle}
            onClick={() => setShowDetail((open) => !open)}
            aria-expanded={showDetail}
          >
            {showDetail ? 'Hide' : 'Show'} detection detail
          </button>
          {showDetail && <p className={styles.reason}>{context.reason}</p>}
        </>
      )}
    </div>
  )
}

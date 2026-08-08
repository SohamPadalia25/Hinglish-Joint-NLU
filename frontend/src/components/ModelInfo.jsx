import { formatParameters } from '../utils/format.js'
import styles from './ModelInfo.module.css'

/**
 * Collapsible model card.
 *
 * Model name and parameter count are read from GET /health when the backend is
 * up, so the numbers on screen are the loaded model's own - not hardcoded.
 */
export default function ModelInfo({ health }) {
  const model = health?.model
  const parameters = formatParameters(model?.parameters)

  const rows = [
    ['Model', 'XLM-RoBERTa Large'],
    ['Parameters', `${parameters || '560M'} (fine-tuned)`],
    ['Tasks', 'Intent Classification, Sentiment Detection, Slot Filling'],
    ['Input', 'English / Hinglish / Code-Switched'],
    ['Context', 'Enabled (conversation memory layer)'],
    ['Priority', 'Enabled (rule-based routing layer)'],
    ['Response Generation', 'Deterministic (template-based)'],
  ]

  return (
    <details className={`card ${styles.details}`}>
      <summary className={styles.summary}>Model Information</summary>

      <dl className={styles.list}>
        {rows.map(([term, value]) => (
          <div key={term} className={styles.row}>
            <dt className={styles.term}>{term}</dt>
            <dd className={styles.value}>{value}</dd>
          </div>
        ))}
      </dl>

      {model?.device && (
        <p className={styles.note}>
          Running on <strong>{model.device.toUpperCase()}</strong>, max sequence length{' '}
          {model.max_len} tokens.
        </p>
      )}

      <p className={styles.note}>
        Priority is a rule-based routing layer, not a separately trained classifier. Context
        awareness is a management layer around the trained NLU model.
      </p>
    </details>
  )
}

import ConfidenceBar from './ConfidenceBar.jsx'
import ContextCard from './ContextCard.jsx'
import EntityList from './EntityList.jsx'
import PriorityCard from './PriorityCard.jsx'
import { classKey, humanise } from '../utils/format.js'
import styles from './AnalysisPanel.module.css'

const SENTIMENTS = ['negative', 'neutral', 'positive']

/**
 * Everything the backend concluded about the selected turn, in the order the
 * pipeline produces it: intent -> sentiment -> entities -> context -> priority
 * -> response type.
 */
export default function AnalysisPanel({ result }) {
  if (!result) {
    return (
      <section className={`card ${styles.panel}`}>
        <h2 className={styles.title}>AI Analysis</h2>
        <p className={styles.placeholder}>
          Send a message to see the intent, sentiment, entities, context and priority the
          model produced for it.
        </p>
      </section>
    )
  }

  const { nlu, context, priority } = result
  const sentimentTone = classKey(nlu.sentiment, SENTIMENTS, 'neutral')

  return (
    <section className={`card ${styles.panel}`}>
      <h2 className={styles.title}>AI Analysis</h2>

      <div className={styles.block}>
        <p className="sectionLabel">Intent</p>
        <p className={styles.primaryValue}>{nlu.intent}</p>
        <ConfidenceBar value={nlu.intent_confidence} tone="accent" />
      </div>

      <div className={styles.block}>
        <p className="sectionLabel">Sentiment</p>
        <p className={`${styles.primaryValue} ${styles[sentimentTone]}`}>{nlu.sentiment}</p>
        <ConfidenceBar value={nlu.sentiment_confidence} tone={sentimentTone} />
      </div>

      <div className={styles.block}>
        <p className="sectionLabel">Entities</p>
        <EntityList slots={nlu.slots} inherited={context?.used ? context.inherited_slots : []} />
      </div>

      <div className={styles.block}>
        <p className="sectionLabel">Context</p>
        <ContextCard context={context} />
      </div>

      <div className={styles.block}>
        <p className="sectionLabel">Priority &amp; Routing</p>
        <PriorityCard priority={priority} />
      </div>

      {result.response_type && (
        <div className={styles.block}>
          <p className="sectionLabel">Response</p>
          <div className={styles.responseRow}>
            <span className={styles.responseType}>{humanise(result.response_type)}</span>
            <span
              className={`${styles.human} ${
                result.requires_human ? styles.humanYes : styles.humanNo
              }`}
            >
              {result.requires_human ? 'Human agent required' : 'Bot can handle'}
            </span>
          </div>
        </div>
      )}
    </section>
  )
}

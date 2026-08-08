import styles from './ExampleQueries.module.css'

// Each one is sent to POST /api/chat exactly as written - no canned replies.
const EXAMPLES = [
  { text: 'Mera order #45231 5 din se deliver nahi hua', note: 'Delivery complaint' },
  { text: '200 rupaye zyada charge kar liye please refund karo', note: 'Refund request' },
  {
    text: 'Complaint 2 hafte se pending hai koi sun nahi raha escalate karo',
    note: 'Escalation',
  },
  { text: 'Thanks for quick help', note: 'Positive feedback' },
  { text: 'Bahut hi worst experience raha', note: 'Negative feedback' },
  { text: 'Maine order #45231 cancel kiya tha', note: 'Cancellation' },
]

export default function ExampleQueries({ onPick, onContextDemo, disabled }) {
  return (
    <div className={`card ${styles.wrapper}`}>
      <div className={styles.head}>
        <h2 className={styles.heading}>Try an example</h2>
        <button
          type="button"
          className={styles.demoButton}
          onClick={onContextDemo}
          disabled={disabled}
          title="Sends two turns in the same session to demonstrate context carry-over"
        >
          Try Context Demo
        </button>
      </div>

      <div className={styles.grid}>
        {EXAMPLES.map((example) => (
          <button
            key={example.text}
            type="button"
            className={styles.example}
            onClick={() => onPick(example.text)}
            disabled={disabled}
          >
            <span className={styles.note}>{example.note}</span>
            <span className={styles.text}>{example.text}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

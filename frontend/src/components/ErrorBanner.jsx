import styles from './ErrorBanner.module.css'

/**
 * User-facing failure message.
 *
 * `api.js` converts every failure into a plain sentence, so nothing technical -
 * status codes, stack traces, exception text - reaches this component.
 */
export default function ErrorBanner({ message, onDismiss }) {
  if (!message) return null

  return (
    <div className={styles.banner} role="alert">
      <div className={styles.body}>
        <span className={styles.title}>Could not complete that request</span>
        <span className={styles.message}>{message}</span>
      </div>
      <button
        type="button"
        className={styles.dismiss}
        onClick={onDismiss}
        aria-label="Dismiss error"
      >
        &times;
      </button>
    </div>
  )
}

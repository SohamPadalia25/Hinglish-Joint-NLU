import { useState } from 'react'

import styles from './ChatInput.module.css'

export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState('')

  const submit = async (event) => {
    // Without this the form would navigate and reload the page.
    event.preventDefault()
    if (disabled) return

    const text = value.trim()
    if (!text) return

    setValue('')
    const sent = await onSend(text)
    // Put the text back so nothing is lost when the backend rejects it.
    if (!sent) setValue(text)
  }

  const handleKeyDown = (event) => {
    // Enter sends; Shift+Enter is a newline.
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit(event)
    }
  }

  return (
    <form className={`card ${styles.form}`} onSubmit={submit}>
      <label className="srOnly" htmlFor="chat-input">
        Your message
      </label>
      <textarea
        id="chat-input"
        className={styles.input}
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Type a Hinglish support query, e.g. Mera order #45231 5 din se deliver nahi hua"
        rows={2}
        disabled={disabled}
        autoComplete="off"
      />
      <div className={styles.actions}>
        <span className={styles.hint}>Enter to send &middot; Shift + Enter for a new line</span>
        <button type="submit" className={styles.send} disabled={disabled || !value.trim()}>
          {disabled ? 'Analysing...' : 'Send'}
        </button>
      </div>
    </form>
  )
}

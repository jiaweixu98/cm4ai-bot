export default function ChatStartup({ error, onRetry }) {
  return (
    <main className="chat-startup" aria-busy={!error}>
      <aside className="chat-startup-sidebar" aria-label="Chat history">
        <strong>Ask TalentKG</strong>
        <div className="chat-startup-placeholder" />
        <div className="chat-startup-placeholder short" />
      </aside>
      <section className="chat-startup-main">
        <header>Ask TalentKG</header>
        <div className="chat-startup-status" role={error ? "alert" : "status"}>
          <h1>{error ? "Your chat isn't ready yet" : "Preparing your chat"}</h1>
          <p>{error || "Loading your conversations and research context…"}</p>
          {error && <button type="button" onClick={onRetry}>Retry loading</button>}
        </div>
        <div className="chat-startup-composer">
          <textarea disabled aria-label="Message (available after loading)" placeholder="Your message" />
          <button disabled>Send</button>
        </div>
      </section>
    </main>
  );
}

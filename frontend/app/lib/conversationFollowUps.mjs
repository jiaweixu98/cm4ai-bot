const key = text => String(text).normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim();

// Empty model output means no suggestions. Never recycle opening cards or
// guesses about what the user should answer to a clarification question.
export function conversationFollowUps({ suggestions = [], messages = [], starters = [], busy = false } = {}) {
  if (busy || messages.at(-1)?.role !== 'assistant') return [];
  const seen = new Set([...messages.map(m => key(m.content || '')), ...starters.map(s => key(s.prompt))]);
  return (Array.isArray(suggestions) ? suggestions : []).flatMap(value => {
    if (typeof value !== 'string') return [];
    const text = value.replace(/\s+/g, ' ').trim();
    const normalized = key(text);
    if (!normalized || text.length > 120 || seen.has(normalized)) return [];
    seen.add(normalized);
    return [text];
  }).slice(0, 3);
}

import { paperTitle } from "./personaConfig";

function splitKeep(text, regex) {
  return String(text || "").split(regex).filter((part) => part !== "");
}

export function FormattedText({ content, citations = [] }) {
  const text = String(content || "");
  const blocks = text.split(/\n{2,}/);

  return (
    <>
      {blocks.map((block, blockIndex) => {
        const lines = block.split("\n");
        const listItems = lines.filter((line) => /^\s*([-*]|\d+\.)\s+/.test(line));
        if (listItems.length >= 2 && listItems.length === lines.length) {
          return (
            <ul key={blockIndex} className="msg-list">
              {lines.map((line, lineIndex) => (
                <li key={lineIndex}>
                  <InlineText text={line.replace(/^\s*([-*]|\d+\.)\s+/, "")} citations={citations} />
                </li>
              ))}
            </ul>
          );
        }
        return (
          <p key={blockIndex} className="msg-p">
            {lines.map((line, lineIndex) => (
              <span key={lineIndex}>
                {lineIndex > 0 && <br />}
                <InlineText text={line} citations={citations} />
              </span>
            ))}
          </p>
        );
      })}
    </>
  );
}

function InlineText({ text, citations }) {
  const parts = splitKeep(text, /(\*\*[^*]+\*\*|`[^`]+`|\[[0-9]+\])/g);
  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code key={index} className="msg-code">
          {part.slice(1, -1)}
        </code>
      );
    }
    const cite = part.match(/^\[([0-9]+)\]$/);
    if (cite) {
      const n = Number(cite[1]);
      const paper = citations[n - 1];
      const label = paper ? paperTitle(paper) : `Title ${n}`;
      return (
        <abbr key={index} className="msg-cite" title={label}>
          [{n}]
        </abbr>
      );
    }
    return <span key={index}>{part}</span>;
  });
}

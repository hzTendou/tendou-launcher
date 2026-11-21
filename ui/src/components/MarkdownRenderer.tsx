import React, { useState } from 'react';
import { Check, Copy } from 'lucide-react';

interface MarkdownRendererProps {
  content: string;
  streaming?: boolean;
}

function renderInline(text: string): React.ReactNode[] {
  // Handle inline code, bold, italic, and links safely without dangerouslySetInnerHTML
  const tokens = text.split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*\n]+\*|\[[^\]]+\]\(https?:\/\/[^)]+\))/g);
  return tokens.map((tok, i) => {
    if (!tok) return null;
    if (tok.startsWith('`') && tok.endsWith('`') && tok.length >= 2) {
      return (
        <code
          key={i}
          className="inline-block px-1.5 py-[1px] rounded-[5px] text-[12px] leading-[1.45] align-baseline font-mono-num whitespace-nowrap max-w-full overflow-x-auto"
          style={{
            background: 'var(--td-surface-3)',
            color: 'var(--td-ink)',
            border: '1px solid var(--td-border)',
          }}
        >
          {tok.slice(1, -1)}
        </code>
      );
    }
    if (tok.startsWith('**') && tok.endsWith('**') && tok.length >= 4) {
      return (
        <strong key={i} className="font-semibold" style={{ color: 'var(--td-ink)' }}>
          {tok.slice(2, -2)}
        </strong>
      );
    }
    if (tok.startsWith('*') && tok.endsWith('*') && tok.length >= 2) {
      return (
        <em key={i} className="italic">
          {tok.slice(1, -1)}
        </em>
      );
    }
    const linkMatch = tok.match(/^\[([^\]]+)\]\((https?:\/\/[^)]+)\)$/);
    if (linkMatch) {
      return (
        <a
          key={i}
          href={linkMatch[2]}
          target="_blank"
          rel="noopener noreferrer"
          className="underline underline-offset-2 hover:opacity-80"
          style={{ color: 'var(--td-blue)' }}
        >
          {linkMatch[1]}
        </a>
      );
    }
    return <React.Fragment key={i}>{tok}</React.Fragment>;
  });
}

function CodeBlock({ lang, code }: { lang: string; code: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    void navigator.clipboard?.writeText(code).catch(() => {});
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };

  return (
    <div
      className="my-3.5 rounded-2xl overflow-hidden"
      style={{
        background: 'var(--td-code-bg)',
        border: '1px solid var(--td-border)',
      }}
    >
      <div
        className="px-4 py-2 flex items-center justify-between text-[11px]"
        style={{
          borderBottom: '1px solid var(--td-border)',
          background: 'var(--td-surface-2)',
          color: 'var(--td-ink-3)',
        }}
      >
        <span className="font-mono-num uppercase tracking-wider text-[10px] font-medium">
          {lang || 'code'}
        </span>
        <button
          type="button"
          onClick={handleCopy}
          className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md transition-colors cursor-pointer hover:opacity-90"
          style={{ color: copied ? 'var(--td-green)' : 'var(--td-ink-2)' }}
        >
          {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>
      <pre
        className="p-4 overflow-x-auto text-[12.5px] leading-[1.6] font-mono-num"
        style={{ color: 'var(--td-ink)' }}
      >
        <code>{code}</code>
      </pre>
    </div>
  );
}

function renderProseBlock(block: string, blockKey: number) {
  const lines = block.split('\n');
  const elements: React.ReactNode[] = [];
  let idx = 0;

  while (idx < lines.length) {
    const line = lines[idx];
    const trimmed = line.trim();

    if (!trimmed) {
      idx += 1;
      continue;
    }

    // Headings
    const hMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (hMatch) {
      const level = hMatch[1].length;
      const cls =
        level === 1
          ? 'text-lg font-semibold font-display mt-4 mb-2'
          : level === 2
            ? 'text-base font-semibold font-display mt-3.5 mb-1.5'
            : 'text-sm font-semibold mt-3 mb-1';
      elements.push(
        <div key={`${blockKey}-h-${idx}`} className={cls} style={{ color: 'var(--td-ink)' }}>
          {renderInline(hMatch[2])}
        </div>,
      );
      idx += 1;
      continue;
    }

    // Horizontal rule
    if (/^[-*_]{3,}$/.test(trimmed)) {
      elements.push(
        <hr key={`${blockKey}-hr-${idx}`} className="my-4" style={{ borderColor: 'var(--td-border)' }} />,
      );
      idx += 1;
      continue;
    }

    // Blockquote
    if (trimmed.startsWith('>')) {
      const quoteLines: string[] = [];
      while (idx < lines.length && lines[idx].trim().startsWith('>')) {
        quoteLines.push(lines[idx].trim().replace(/^>\s?/, ''));
        idx += 1;
      }
      elements.push(
        <blockquote
          key={`${blockKey}-q-${idx}`}
          className="my-2.5 pl-3.5 py-0.5 text-[13.5px] leading-relaxed"
          style={{
            borderLeft: '2px solid var(--td-accent)',
            color: 'var(--td-ink-2)',
          }}
        >
          {quoteLines.map((ql, qi) => (
            <p key={qi}>{renderInline(ql)}</p>
          ))}
        </blockquote>,
      );
      continue;
    }

    // Simple Markdown table
    if (
      trimmed.includes('|') &&
      idx + 1 < lines.length &&
      /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[idx + 1])
    ) {
      const parseRow = (r: string) =>
        r
          .trim()
          .replace(/^\|/, '')
          .replace(/\|$/, '')
          .split('|')
          .map((c) => c.trim());
      const headers = parseRow(line);
      idx += 2;
      const rows: string[][] = [];
      while (idx < lines.length && lines[idx].includes('|') && lines[idx].trim() !== '') {
        rows.push(parseRow(lines[idx]));
        idx += 1;
      }
      elements.push(
        <div
          key={`${blockKey}-tbl-${idx}`}
          className="my-3 overflow-x-auto rounded-xl"
          style={{ border: '1px solid var(--td-border)' }}
        >
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr style={{ background: 'var(--td-surface-2)', borderBottom: '1px solid var(--td-border)' }}>
                {headers.map((h, hi) => (
                  <th key={hi} className="px-3 py-2 font-semibold" style={{ color: 'var(--td-ink)' }}>
                    {renderInline(h)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, ri) => (
                <tr key={ri} style={{ borderBottom: ri < rows.length - 1 ? '1px solid var(--td-border)' : 'none' }}>
                  {headers.map((_, ci) => (
                    <td key={ci} className="px-3 py-2" style={{ color: 'var(--td-ink-2)' }}>
                      {renderInline(row[ci] ?? '')}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }

    // Unordered list
    if (/^[-*+]\s+/.test(trimmed)) {
      const items: string[] = [];
      while (idx < lines.length && /^\s*[-*+]\s+/.test(lines[idx])) {
        items.push(lines[idx].replace(/^\s*[-*+]\s+/, ''));
        idx += 1;
      }
      elements.push(
        <ul key={`${blockKey}-ul-${idx}`} className="my-2 pl-5 list-disc space-y-1">
          {items.map((it, ii) => (
            <li key={ii} className="leading-[1.65]">
              {renderInline(it)}
            </li>
          ))}
        </ul>,
      );
      continue;
    }

    // Ordered list
    if (/^\d+\.\s+/.test(trimmed)) {
      const items: string[] = [];
      while (idx < lines.length && /^\s*\d+\.\s+/.test(lines[idx])) {
        items.push(lines[idx].replace(/^\s*\d+\.\s+/, ''));
        idx += 1;
      }
      elements.push(
        <ol key={`${blockKey}-ol-${idx}`} className="my-2 pl-5 list-decimal space-y-1">
          {items.map((it, ii) => (
            <li key={ii} className="leading-[1.65]">
              {renderInline(it)}
            </li>
          ))}
        </ol>,
      );
      continue;
    }

    // Regular paragraph (group consecutive plain lines)
    const paraLines: string[] = [line];
    idx += 1;
    while (
      idx < lines.length &&
      lines[idx].trim() !== '' &&
      !/^(#{1,4}\s+|[-*+]\s+|\d+\.\s+|>\s?|[-*_]{3,}$)/.test(lines[idx].trim()) &&
      !(
        lines[idx].includes('|') &&
        idx + 1 < lines.length &&
        /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[idx + 1])
      )
    ) {
      paraLines.push(lines[idx]);
      idx += 1;
    }
    elements.push(
      <p key={`${blockKey}-p-${idx}`} className="my-1.5 leading-[1.7] whitespace-pre-wrap break-words">
        {renderInline(paraLines.join('\n'))}
      </p>,
    );
  }

  return elements;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, streaming = false }) => {
  // Support both closed fenced blocks and an in-progress open code fence during streaming
  const segments: Array<{ kind: 'text' | 'code'; lang?: string; body: string }> = [];
  const fenceRe = /```([a-zA-Z0-9_+#.-]*)\n?([\s\S]*?)(?:```|$)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = fenceRe.exec(content)) !== null) {
    if (match.index > lastIndex) {
      segments.push({ kind: 'text', body: content.slice(lastIndex, match.index) });
    }
    segments.push({ kind: 'code', lang: match[1] || 'code', body: match[2].replace(/\n$/, '') });
    lastIndex = fenceRe.lastIndex;
  }
  if (lastIndex < content.length) {
    segments.push({ kind: 'text', body: content.slice(lastIndex) });
  }

  if (segments.length === 0 && streaming) {
    return (
      <div className="py-1 flex items-center gap-1.5 text-sm" style={{ color: 'var(--td-ink-3)' }}>
        <span className="inline-block w-1.5 h-4 rounded-full td-stream-cursor" style={{ background: 'var(--td-accent)' }} />
      </div>
    );
  }

  return (
    <div className="text-[14.5px]" style={{ color: 'var(--td-ink)' }}>
      {segments.map((seg, i) =>
        seg.kind === 'code' ? (
          <CodeBlock key={i} lang={seg.lang ?? 'code'} code={seg.body} />
        ) : (
          <React.Fragment key={i}>{renderProseBlock(seg.body, i)}</React.Fragment>
        ),
      )}
      {streaming && (
        <span
          className="inline-block w-1.5 h-4 ml-1 align-middle rounded-full td-stream-cursor"
          style={{ background: 'var(--td-accent)' }}
        />
      )}
    </div>
  );
};

const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const postsDir = path.join(root, "posts");

const escapeHtml = (value) =>
  value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

const renderInline = (value) => {
  let html = escapeHtml(value);

  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  html = html.replace(
    /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
    '<a href="$2" target="_blank" rel="noreferrer">$1</a>'
  );
  html = html.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2">$1</a>');

  return html;
};

const renderTable = (lines) => {
  const rows = lines
    .filter((line) => line.trim())
    .map((line) =>
      line
        .trim()
        .replace(/^\|/, "")
        .replace(/\|$/, "")
        .split("|")
        .map((cell) => cell.trim())
    );

  if (rows.length < 2) {
    return "";
  }

  const headers = rows[0]
    .map((cell) => `<th>${renderInline(cell)}</th>`)
    .join("");
  const bodyRows = rows
    .slice(2)
    .map(
      (row) =>
        `<tr>${row
          .map((cell) => `<td>${renderInline(cell)}</td>`)
          .join("")}</tr>`
    )
    .join("");

  return `<div class="markdown-table-wrap"><table><thead><tr>${headers}</tr></thead><tbody>${bodyRows}</tbody></table></div>`;
};

const markdownToHtml = (markdown) => {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let paragraph = [];
  let list = null;
  let blockquote = [];
  let code = null;
  let table = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    html.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
    paragraph = [];
  };

  const flushList = () => {
    if (!list) return;
    const startAttr =
      list.type === "ol" && list.start && list.start !== "1"
        ? ` start="${list.start}"`
        : "";
    html.push(
      `<${list.type}${startAttr}>${list.items
        .map((item) => `<li>${renderInline(item)}</li>`)
        .join("")}</${list.type}>`
    );
    list = null;
  };

  const flushBlockquote = () => {
    if (!blockquote.length) return;
    html.push(
      `<blockquote>${blockquote
        .map((line) => `<p>${renderInline(line)}</p>`)
        .join("")}</blockquote>`
    );
    blockquote = [];
  };

  const flushTable = () => {
    if (!table.length) return;
    html.push(renderTable(table));
    table = [];
  };

  const flushOpenBlocks = () => {
    flushParagraph();
    flushList();
    flushBlockquote();
    flushTable();
  };

  lines.forEach((line) => {
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {
      if (code) {
        html.push(`<pre><code>${escapeHtml(code.lines.join("\n"))}</code></pre>`);
        code = null;
      } else {
        flushOpenBlocks();
        code = { lines: [] };
      }
      return;
    }

    if (code) {
      code.lines.push(line);
      return;
    }

    if (!trimmed) {
      flushOpenBlocks();
      return;
    }

    if (trimmed.includes("|") && /^\|?[-:\s|]+\|[-:\s|]+$/.test(trimmed)) {
      table.push(trimmed);
      return;
    }

    if (trimmed.includes("|") && (table.length || /^\|.+\|$/.test(trimmed))) {
      flushParagraph();
      flushList();
      flushBlockquote();
      table.push(trimmed);
      return;
    }

    flushTable();

    if (/^---+$/.test(trimmed)) {
      flushOpenBlocks();
      html.push("<hr />");
      return;
    }

    const heading = /^(#{1,6})\s+(.+)$/.exec(trimmed);
    if (heading) {
      flushOpenBlocks();
      const level = heading[1].length;
      html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      return;
    }

    if (trimmed.startsWith(">")) {
      flushParagraph();
      flushList();
      blockquote.push(trimmed.replace(/^>\s?/, ""));
      return;
    }

    const unordered = /^[-*]\s+(.+)$/.exec(trimmed);
    if (unordered) {
      flushParagraph();
      flushBlockquote();
      if (!list || list.type !== "ul") {
        flushList();
        list = { type: "ul", items: [] };
      }
      list.items.push(unordered[1]);
      return;
    }

    const ordered = /^(\d+)\.\s+(.+)$/.exec(trimmed);
    if (ordered) {
      flushParagraph();
      flushBlockquote();
      if (!list || list.type !== "ol") {
        flushList();
        list = { type: "ol", items: [], start: ordered[1] };
      }
      list.items.push(ordered[2]);
      return;
    }

    flushList();
    flushBlockquote();
    paragraph.push(trimmed);
  });

  if (code) {
    html.push(`<pre><code>${escapeHtml(code.lines.join("\n"))}</code></pre>`);
  }
  flushOpenBlocks();

  return html.join("\n");
};

const titlePattern = /^#\s+(.+)$/m;

const postShell = ({ title, body }) => `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${escapeHtml(title)} — Jaeho Jung</title>
    <link rel="stylesheet" href="../css/main.css" />
    <link rel="stylesheet" href="../css/media.css" />
  </head>
  <body class="site">
    <header class="header">
      <div class="wrap header-wrap">
        <a href="../index.html#home" class="logo">Jaeho Jung</a>
        <nav class="nav" aria-label="Primary navigation">
          <a href="../index.html#about" class="nav-link">About</a>
          <a href="../index.html#projects" class="nav-link">Research</a>
          <a href="index.html" class="nav-link active">Writing</a>
          <a href="../index.html#performance" class="nav-link">Performance</a>
          <a href="../index.html#contact" class="nav-link">Contact</a>
        </nav>
      </div>
    </header>

    <main>
      <article class="post-page wrap">
        <div class="post-kicker">Jazz Is a Language</div>
        <div class="post-content">
${body}
        </div>
        <nav class="post-nav" aria-label="Post navigation">
          <a href="index.html">All posts</a>
          <a href="../index.html#writing">Back to home</a>
        </nav>
      </article>
    </main>
  </body>
</html>
`;

const files = fs
  .readdirSync(postsDir)
  .filter((file) => file.endsWith(".md"))
  .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

files.forEach((file) => {
  const markdown = fs.readFileSync(path.join(postsDir, file), "utf8").trim();
  const title = markdown.match(titlePattern)?.[1]?.trim() || path.basename(file, ".md");
  const body = markdownToHtml(markdown);
  const outputPath = path.join(postsDir, file.replace(/\.md$/, ".html"));

  fs.writeFileSync(outputPath, postShell({ title, body }));
  console.log(`Generated ${path.relative(root, outputPath)}`);
});

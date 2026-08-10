# Jaeho Jung

Personal website for Jaeho Jung.

The site presents selected AI work, writing on controllable music generation,
guitar performance links, and contact information. It is intentionally kept as
a small static site: HTML, CSS, and a little JavaScript.

## Structure

- `index.html` - main personal website
- `css/` - site styles
- `js/` - navigation behavior
- `posts/` - generated blog pages and source Markdown posts
- `scripts/build-post-pages.js` - local post-page generator
- `img/` - site favicon and profile photo

## Local Preview

Open `index.html` directly in a browser, or serve the directory with any static
file server.

## Blog Pages

After editing Markdown files in `posts/`, regenerate the HTML pages:

```bash
node scripts/build-post-pages.js
```

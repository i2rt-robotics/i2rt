# I2RT Robotics API — Documentation Site

A static documentation website for the [i2rt](../) codebase, covering the YAM
arms and the Flow Base mobile base from the ground up.

## View it

- **Hosted (GitHub Pages):** enabled via `.github/workflows/deploy-docs.yml`
  (Settings → Pages → Source: *GitHub Actions*).
- **Locally:** just open `site/index.html` in a browser, or serve the folder:
  ```bash
  python -m http.server -d site 8000   # then visit http://localhost:8000
  ```

## Files

| File | Purpose |
|------|---------|
| `index.html` | The full single-page documentation |
| `styles.css` | Styling (dark/light theme) |
| `script.js`  | Sidebar scrollspy, search, theme toggle, copy buttons |

No build step and no dependencies — pure HTML/CSS/JS.

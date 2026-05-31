# Deploy checklist: dashboard → robinwyeo.github.io

Goal: run the Streamlit dashboard on a free host, then surface it as a post on the
**(Data) Science** page of `robinwyeo.github.io` via an embedded iframe + link.

> Why not host it directly on GitHub Pages? GitHub Pages serves **static files only**.
> This app is a Python/Streamlit server (it runs `yfinance`, scoring, etc. on each
> request), so it must run on a host that executes Python. The website just embeds it.

---

## Part 1 — Deploy the Streamlit app (Streamlit Community Cloud, free)

- [ ] **Push `financial-tools` to GitHub** (public repo, branch `main`).
- [ ] Confirm `requirements.txt` is committed and complete (it is: streamlit, yfinance, pandas, etc.).
- [ ] Commit an initial universe snapshot so cross-sectional ranks work on first load:
      `python -m core.universe --fast --max 50` → ensure `data/universe_snapshot.parquet` is committed.
      (Check `.gitignore` isn't excluding it; if it is, force-add with `git add -f data/universe_snapshot.parquet`.)
- [ ] Go to **[share.streamlit.io](https://share.streamlit.io)** → sign in with GitHub → **New app**.
- [ ] Select repo `robinwyeo/financial-tools`, branch `main`, main file **`app.py`** → Deploy.
- [ ] Wait for the build; note the public URL (looks like `https://<app-name>.streamlit.app`).

### Secrets / config (only if needed)
- [ ] The **dashboard itself needs no secrets**. The email/SMTP settings in `config.yaml`
      and the daily alert job are for the GitHub Actions workflow, not the public app.
- [ ] If you later want secrets in the app, add them under the app's **Settings → Secrets**
      on Streamlit Cloud (never commit credentials).

### Allow embedding in an iframe
- [ ] Streamlit Cloud apps support embed mode via the `?embed=true` query param (already used
      in the post). No extra config needed for a basic embed.
- [ ] Test the embed URL directly in a browser: `https://<app-name>.streamlit.app/?embed=true`

---

## Part 2 — Wire it into the website (`robinwyeo.github.io`)

The post file is already scaffolded at:
`_data_science/2026-05-30-financial-tools.md`

- [ ] **Replace the placeholder URL.** In that post, swap every `YOUR-APP-NAME.streamlit.app`
      for your real Streamlit URL (appears 2×: iframe `src` and the "open in new tab" link).
- [x] **Teaser/title image** added at `images/data-science/financial-tools/title.png`
      (a generated dashboard mockup). Swap in a real screenshot later if you prefer.
- [ ] (Optional) Remove the top `<!-- TODO ... -->` comment block from the post once the URL is set.
- [ ] **Preview locally** (Jekyll):
      ```bash
      cd robinwyeo.github.io
      bundle exec jekyll serve
      ```
      Visit `http://localhost:4000/data-science/` — the new post should appear under **2026**,
      then open `http://localhost:4000/data-science/financial-tools/` and confirm the dashboard
      embeds and the "open in new tab" link works.
- [ ] **Commit & push** the website repo. GitHub Pages rebuilds automatically; the post goes
      live at `https://robinwyeo.github.io/data-science/financial-tools/`.

---

## Notes & gotchas

- **Cold starts:** free Streamlit apps sleep when idle and take ~30s to wake. The post already
  warns readers about this and offers an "open in new tab" link.
- **iframe height/mobile:** the embed is a fixed 900px tall. If it feels cramped, adjust the
  `height` in the iframe `style`, or lean on the "open in new tab" link as the primary CTA.
- **Data freshness:** `yfinance` is a free, occasionally-flaky source. To keep universe ranks
  fresh, the existing `.github/workflows/daily.yml` job rebuilds the snapshot; make sure that
  workflow is enabled on the GitHub repo if you want daily refreshes.
- **No investment advice:** the post includes a disclaimer — keep it.

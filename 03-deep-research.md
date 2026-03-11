# Deep Research — Complete Standalone Agent Prompt

## Project Identity

| Field | Value |
|-------|-------|
| **Project Folder** | `C:\Users\namsi\Desktop\Projects\Deep Resesarch` *(note: folder has a typo — "Resesarch")* |
| **Tech Stack** | Python/Flask backend + Vanilla JS frontend |
| **Vercel URL** | https://deep-research-zeta-sage.vercel.app/ |
| **GitHub Repo** | `NamanSingh69/Deep-Research` (already exists) |
| **Vercel Env Vars** | `GEMINI_API_KEY` is set |

### Key Files
- `app.py` — Flask backend with research endpoints
- `templates/index.html` — Main HTML template (Jinja2)
- `static/` — CSS, JS, and static assets
- `static/gemini-client.js` — Should be v2 (28KB) with Pro/Fast toggle, rate limit counter, model cascade
- `api/` — Vercel Serverless Functions
- `vercel.json` — Route configuration mapping `/api/*` to Python functions
- `requirements.txt` — Python dependencies

---

## Shared Infrastructure Context (CRITICAL — Read Before Making Changes)

This project is part of a 16-project portfolio. Previous work sessions established shared patterns you MUST follow:

### Design System — The "UX Mandate"
All projects must implement **4 core UI states**:
1. **Loading** → Animated skeleton screen placeholders (shimmer effect)
2. **Success** → Toast notifications (CSS-animated, green, auto-dismiss after 4s)
3. **Empty** → Beautiful null states with friendly messaging
4. **Error** → Red toast with actionable recovery

**NEVER use native `alert()`. Replace ALL with toast notifications.**

### Gemini Client (Python/Vanilla JS projects)
The standard client is `gemini-client.js` v2 (28KB). It provides:
- **Pro/Fast Toggle**: Floating `⚡ PRO / 🚀 FAST` pill button, persists in `localStorage.gemini_mode`
- **Rate Limit Counter**: Visual `Requests: X/15 remaining` badge
- **Model Cascade**: Auto-fallback on 429/503 errors
### Smart Model Cascade (March 2026)
**Primary (Free Preview):** `gemini-3.1-pro-preview` → `gemini-3-flash-preview` → `gemini-3.1-flash-lite-preview`
**Fallback (Free Stable):** `gemini-2.5-pro` → `gemini-2.5-flash` → `gemini-2.5-flash-lite`
**Note:** `gemini-2.0-*` deprecated March 3, 2026. Do NOT use.
**Grounding:** This is a **real-time data project** (web research) → use `gemini-2.5-pro` or `gemini-2.5-flash` with Google Search grounding (5K free queries/month)
- Config: `window.GEMINI_CONFIG = { needsRealTimeData: true }` — prefers `gemini-2.5-pro` first

### Security
- Backend must use `os.environ.get("GEMINI_API_KEY")` — never hardcode
- `.gitignore` covers `.env*`, `node_modules/`, `.vercel/`, `__pycache__/`

### Mobile Responsiveness (Required)
- `<meta name="viewport" content="width=device-width, initial-scale=1.0">` in `<head>`
- All layouts must work 375px–1920px
- Touch targets ≥ 44×44px
- No horizontal scrolling

---

## Current Live State (Verified March 10, 2026)

| Feature | Status | Details |
|---------|--------|---------|
| Site loads | ✅ 200 OK | AI Research dashboard with depth/breadth configuration |
| Login wall | ✅ None | No login required |
| Pro/Fast Toggle | ✅ Present | PRO/FAST toggle visible in the UI |
| Rate Limit Counter | ❌ Static only | Shows "Free Tier Limits: 15 Requests/Min..." as static text — NOT a live counter |
| Empty State | ✅ Present | "Welcome to Deep Research Tool! Enter your question to begin." |
| Skeleton Loaders | ❌ MISSING | Shows "Processing..." static text, no animated skeletons |
| Toasts | ❌ MISSING | Status messages appear in status bar or chat bubbles, not as toasts |
| Mobile Responsive | ✅ Yes | Layout adapts at 375px width |
| Console Errors | ⚠️ SyntaxError | `SyntaxError` when fetching previous sessions — API endpoint returning HTML instead of JSON |

---

## Required Changes

### 1. Fix SyntaxError Console Error (HIGH PRIORITY)
The console shows a `SyntaxError` when the page loads — likely the `/api/sessions` or similar endpoint is returning HTML (possibly a Vercel 404 page) instead of JSON.
- Check `vercel.json` routes — ensure the session-loading endpoint is correctly mapped
- Check the Python backend route for loading sessions — it should return `jsonify(...)` not render a template
- If the sessions feature isn't implemented yet, either:
  - Return an empty JSON array `[]` from the endpoint
  - Or remove the client-side fetch call that's failing

### 2. Replace Static Rate Limit Text with Live Counter
The current "Free Tier Limits: 15 Requests/Min..." static text must become a dynamic counter:
- Replace the static text with a live `Requests: X/15 remaining` display
- Decrement on each API call to `/research`
- Store count + reset timestamp in `localStorage`
- Reset after 60 seconds
- Show a warning toast when 2 requests remain
- The `gemini-client.js` v2 already handles this if properly configured — verify it's injected

### 3. Add Animated Skeleton Loaders
When a research query is submitted:
- Replace the "Processing..." static text with animated skeleton placeholders
- Skeletons should mimic the shape of research results (title line, content paragraphs, source links)
- Use shimmer animation matching the app's color scheme:
```css
.research-skeleton {
  background: linear-gradient(90deg, #1e293b 25%, #334155 50%, #1e293b 75%);
  background-size: 200px 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 6px;
}
```

### 4. Add Toast Notification System
- Create a toast system for success/error/info messages:
  - Success toast when research completes: "Research complete! Found X sources."
  - Error toast on failure: "Research failed. Please try again."
  - "Configuration saved" messages → toasts, NOT status bar text
- Implementation: CSS-animated, fixed bottom-right, auto-dismiss 4s

### 5. Verify `/research` Endpoint Works
Previous audits reported a **405 Method Not Allowed** on the `/research` POST endpoint.
- Check `vercel.json` — ensure the route for `/research` maps to the correct Python handler
- Check the Python backend: `@app.route('/research', methods=['POST'])` decorator must be present
- Check that the serverless function wrapper in `api/` correctly exposes this route
- Test locally: `python app.py` then `curl -X POST http://localhost:5000/research -H "Content-Type: application/json" -d '{"query":"test"}'`

### 6. Mobile Responsiveness Hardening
- The research configuration panel (depth/breadth sliders) must be usable on mobile
- Research results should scroll vertically, not overflow horizontally
- Source links should wrap properly on small screens
- The query input area must be easily typeable on mobile keyboards

### 7. GitHub & Deployment
- Push to `Deep-Research` repo (already exists)
- `git add -A && git commit -m "feat: fix session loading, live rate counter, skeletons, toasts, mobile hardening" && git push`
- Redeploy: `npx vercel --prod --yes`
- Verify at https://deep-research-zeta-sage.vercel.app/

---

## Verification Checklist
1. ✅ Page loads — no SyntaxError in console
2. ✅ "Welcome to Deep Research Tool!" empty state displays
3. ✅ Pro/Fast toggle visible and persists selection in `localStorage.gemini_mode`
4. ✅ Rate limit counter shows live count (NOT static "Free Tier Limits" text)
5. ✅ Submit a research query → animated skeleton loaders appear (shimmer, not "Processing...")
6. ✅ Research completes → success toast fires
7. ✅ `/research` POST endpoint returns 200 (not 405)
8. ✅ Resize to 375px → fully usable, no horizontal scroll
9. ✅ DevTools console → zero errors

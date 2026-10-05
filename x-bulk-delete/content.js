// X Bulk Delete: content script.
// Drives x.com's own UI (the "..." menu, then Delete, then Confirm) for every
// post in the timeline that belongs to the logged-in account. Nothing is sent
// anywhere except to x.com itself through its normal interface.

(() => {
  if (window.__xBulkDeleteLoaded) return;
  window.__xBulkDeleteLoaded = true;

  // Menu label for "Delete" in common UI languages. Used together with the
  // menu item's red color so it still works if your language isn't listed.
  const DELETE_LABELS = [
    "delete", "حذف", "supprimer", "eliminar", "löschen", "excluir", "sil",
    "削除", "elimina", "удалить", "verwijderen", "usuń", "삭제", "删除", "刪除",
  ];

  const state = {
    running: false,
    stopRequested: false,
    deleted: 0,
    unreposted: 0,
    skipped: 0,
    errors: 0,
    status: "Idle",
  };

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const jitter = (ms) => ms + Math.floor(Math.random() * ms * 0.5);

  async function waitFor(selectorOrFn, timeout = 4000, root = document) {
    const start = Date.now();
    while (Date.now() - start < timeout) {
      const el =
        typeof selectorOrFn === "function"
          ? selectorOrFn()
          : root.querySelector(selectorOrFn);
      if (el) return el;
      await sleep(100);
    }
    return null;
  }

  // ---------- Status reporting (popup + on-page badge) ----------

  let badge;
  function ensureBadge() {
    if (badge && document.body.contains(badge)) return badge;
    badge = document.createElement("div");
    badge.style.cssText = [
      "position:fixed", "bottom:20px", "left:20px", "z-index:2147483647",
      "background:#0f1419", "color:#fff", "padding:12px 14px",
      "border-radius:12px", "font:13px/1.4 -apple-system,system-ui,sans-serif",
      "box-shadow:0 4px 20px rgba(0,0,0,.35)", "min-width:220px",
    ].join(";");
    const text = document.createElement("div");
    text.id = "xbd-text";
    const stop = document.createElement("button");
    stop.textContent = "Stop";
    stop.style.cssText =
      "margin-top:8px;background:#f4212e;color:#fff;border:0;border-radius:999px;padding:4px 14px;font-weight:600;cursor:pointer";
    stop.onclick = () => (state.stopRequested = true);
    badge.append(text, stop);
    document.body.appendChild(badge);
    return badge;
  }

  function report(status) {
    if (status) state.status = status;
    const snapshot = { ...state };
    try {
      chrome.storage.local.set({ xbdState: snapshot });
      chrome.runtime.sendMessage({ type: "xbd-progress", state: snapshot }).catch(() => {});
    } catch (_) {
      // Popup closed or extension reloaded; the badge still shows progress.
    }
    const b = ensureBadge();
    b.querySelector("#xbd-text").innerHTML =
      `<b>X Bulk Delete</b><br>${escapeHtml(state.status)}<br>` +
      `Deleted: ${state.deleted} · Reposts undone: ${state.unreposted}<br>` +
      `Skipped: ${state.skipped} · Errors: ${state.errors}`;
    b.querySelector("button").style.display = state.running ? "" : "none";
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  // ---------- Reading the page ----------

  function getMyHandle() {
    const link = document.querySelector('a[data-testid="AppTabBar_Profile_Link"]');
    if (!link) return null;
    return link.getAttribute("href").replace(/^\//, "").split(/[/?#]/)[0].toLowerCase();
  }

  function authorHandle(article) {
    const userName = article.querySelector('[data-testid="User-Name"]');
    if (!userName) return null;
    for (const a of userName.querySelectorAll('a[href^="/"]')) {
      const path = a.getAttribute("href").slice(1);
      if (path && !path.includes("/")) return path.toLowerCase();
    }
    return null;
  }

  function postDate(article) {
    const time = article.querySelector("time[datetime]");
    return time ? new Date(time.getAttribute("datetime")) : null;
  }

  function postText(article) {
    const t = article.querySelector('[data-testid="tweetText"]');
    return t ? t.innerText : "";
  }

  function isPinned(article) {
    const ctx = article.querySelector('[data-testid="socialContext"]');
    // The pinned label has no link; the repost label links to a profile.
    return !!ctx && !ctx.closest("a") && !article.querySelector('[data-testid="unretweet"]');
  }

  function matchesFilters(article, opts) {
    const date = postDate(article);
    if (opts.before && date && date >= new Date(opts.before + "T23:59:59")) return false;
    if (opts.after && date && date < new Date(opts.after + "T00:00:00")) return false;
    if (opts.keyword) {
      const words = opts.keyword.toLowerCase().split(",").map((w) => w.trim()).filter(Boolean);
      const text = postText(article).toLowerCase();
      if (words.length && !words.some((w) => text.includes(w))) return false;
    }
    if (opts.keepPinned && isPinned(article)) return false;
    return true;
  }

  // ---------- Actions ----------

  function isDeleteItem(item) {
    const label = item.innerText.trim().toLowerCase();
    if (DELETE_LABELS.some((l) => label === l || label.startsWith(l))) return true;
    const span = item.querySelector("span") || item;
    const [r, g, b] = (getComputedStyle(span).color.match(/\d+/g) || []).map(Number);
    return r > 200 && g < 80 && b < 80; // X renders Delete in red
  }

  async function closeMenus() {
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    await sleep(200);
  }

  async function deletePost(article) {
    const caret = article.querySelector('[data-testid="caret"]');
    if (!caret) return false;
    caret.click();

    const menu = await waitFor('[role="menu"]');
    if (!menu) return false;
    const item = [...menu.querySelectorAll('[role="menuitem"]')].find(isDeleteItem);
    if (!item) {
      await closeMenus();
      return false;
    }
    item.click();

    const confirm = await waitFor('[data-testid="confirmationSheetConfirm"]');
    if (!confirm) {
      await closeMenus();
      return false;
    }
    confirm.click();
    await waitFor(() => !document.body.contains(article) || null, 3000);
    return true;
  }

  async function undoRepost(article) {
    const btn = article.querySelector('[data-testid="unretweet"]');
    if (!btn) return false;
    btn.click();
    const confirm = await waitFor('[data-testid="unretweetConfirm"]');
    if (!confirm) {
      await closeMenus();
      return false;
    }
    confirm.click();
    return true;
  }

  // ---------- Main loop ----------

  async function run(opts) {
    if (state.running) return;
    const me = getMyHandle();
    if (!me) {
      report("Couldn't find your account. Make sure you're logged in to x.com.");
      return;
    }

    Object.assign(state, { running: true, stopRequested: false, deleted: 0, unreposted: 0, skipped: 0, errors: 0 });
    report(`Running as @${me}${opts.dryRun ? " (dry run)" : ""}…`);

    const delay = Math.max(500, Number(opts.delayMs) || 1500);
    const limit = Number(opts.limit) || Infinity;
    let idleScrolls = 0;

    while (!state.stopRequested && state.deleted + state.unreposted < limit) {
      const articles = [...document.querySelectorAll('article[data-testid="tweet"]:not([data-xbd-done])')];
      let acted = false;

      for (const article of articles) {
        if (state.stopRequested || state.deleted + state.unreposted >= limit) break;
        article.setAttribute("data-xbd-done", "1");

        const author = authorHandle(article);
        const reposted = !!article.querySelector('[data-testid="unretweet"]');
        const mine = author === me;

        let action = null;
        if (mine && opts.deletePosts) action = "delete";
        else if (!mine && reposted && opts.undoReposts) action = "unrepost";
        if (!action) continue;

        if (!matchesFilters(article, opts)) {
          state.skipped++;
          continue;
        }

        article.scrollIntoView({ block: "center" });
        await sleep(300);

        if (opts.dryRun) {
          article.style.outline = "3px solid #f4212e";
          action === "delete" ? state.deleted++ : state.unreposted++;
          report(`Dry run: would ${action === "delete" ? "delete" : "undo repost of"} a post`);
          acted = true;
          continue;
        }

        try {
          const ok = action === "delete" ? await deletePost(article) : await undoRepost(article);
          if (ok) {
            action === "delete" ? state.deleted++ : state.unreposted++;
            acted = true;
            report("Working…");
          } else {
            state.errors++;
            report("Couldn't act on one post, moving on.");
          }
        } catch (e) {
          state.errors++;
          report(`Error: ${e.message}`);
          await closeMenus();
        }
        await sleep(jitter(delay));
      }

      if (!acted) {
        // Nothing to do in view: load more of the timeline.
        const before = document.documentElement.scrollHeight;
        window.scrollTo(0, before);
        await sleep(jitter(1500));
        if (document.documentElement.scrollHeight === before) {
          idleScrolls++;
          report(`Looking for more posts… (${idleScrolls}/8)`);
          if (idleScrolls >= 8) break;
        } else {
          idleScrolls = 0;
        }
      } else {
        idleScrolls = 0;
      }
    }

    state.running = false;
    report(state.stopRequested ? "Stopped." : "Done. Reload the page and run again if posts remain.");
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg.type === "xbd-start") {
      run(msg.options);
      sendResponse({ ok: true });
    } else if (msg.type === "xbd-stop") {
      state.stopRequested = true;
      sendResponse({ ok: true });
    } else if (msg.type === "xbd-status") {
      sendResponse({ state: { ...state }, me: getMyHandle(), path: location.pathname });
    }
  });
})();

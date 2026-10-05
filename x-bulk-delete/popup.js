const FIELDS = ["deletePosts", "undoReposts", "keepPinned", "after", "before", "keyword", "limit", "delayMs", "dryRun"];
const $ = (id) => document.getElementById(id);

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

function isX(tab) {
  return tab && /^https:\/\/(x|twitter)\.com\//.test(tab.url || "");
}

function readOptions() {
  const o = {};
  for (const f of FIELDS) {
    const el = $(f);
    o[f] = el.type === "checkbox" ? el.checked : el.value;
  }
  return o;
}

function showState(s) {
  if (!s) return;
  $("status").textContent =
    `${s.status}\nDeleted: ${s.deleted} · Reposts undone: ${s.unreposted}\nSkipped: ${s.skipped} · Errors: ${s.errors}`;
}

async function send(msg) {
  const tab = await activeTab();
  if (!isX(tab)) {
    $("status").textContent = "Open x.com in this tab first.";
    return null;
  }
  try {
    return await chrome.tabs.sendMessage(tab.id, msg);
  } catch {
    // Content script not injected yet (tab was open before install): inject it.
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] });
    return chrome.tabs.sendMessage(tab.id, msg);
  }
}

async function goTo(suffix) {
  const res = await send({ type: "xbd-status" });
  if (!res || !res.me) {
    $("status").textContent = "Log in to x.com first.";
    return;
  }
  const tab = await activeTab();
  chrome.tabs.update(tab.id, { url: `https://x.com/${res.me}${suffix}` });
  window.close();
}

async function init() {
  // Restore saved options.
  const { xbdOptions, xbdState } = await chrome.storage.local.get(["xbdOptions", "xbdState"]);
  if (xbdOptions) {
    for (const f of FIELDS) {
      if (!(f in xbdOptions)) continue;
      const el = $(f);
      el.type === "checkbox" ? (el.checked = xbdOptions[f]) : (el.value = xbdOptions[f]);
    }
  }
  showState(xbdState);

  const res = await send({ type: "xbd-status" }).catch(() => null);
  if (res && res.me) {
    $("account").textContent = `Logged in as @${res.me}`;
    showState(res.state);
  }

  $("goPosts").onclick = () => goTo("");
  $("goReplies").onclick = () => goTo("/with_replies");

  let armed = false;
  $("start").onclick = async () => {
    const options = readOptions();
    await chrome.storage.local.set({ xbdOptions: options });
    // Real deletes need a second click to confirm.
    if (!options.dryRun && !armed) {
      armed = true;
      $("start").textContent = "Click again to delete";
      $("status").textContent = "This permanently deletes posts from your X account.";
      return;
    }
    armed = false;
    $("start").textContent = "Start";
    await send({ type: "xbd-start", options });
  };
  $("stop").onclick = () => send({ type: "xbd-stop" });
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "xbd-progress") showState(msg.state);
});

init();

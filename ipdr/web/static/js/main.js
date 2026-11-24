
import { aiReady, api, state } from "./api.js";
import { currentModelLabel, initChat, refreshVoice, showChat } from "./chat.js";
import { renderDictionary } from "./dictionary.js";
import { renderExplore } from "./explore.js";
import { icon } from "./icons.js";
import { onSettingsSaved, renderSettings } from "./settings.js";
import { renderTerms } from "./terms.js";
import { initTheme } from "./theme.js";
import { cancelAllVoice } from "./voice.js";
import { $, $$, fmtDay, toast } from "./util.js";

const VIEWS = ["chat", "explore", "dictionary", "terms", "settings"];

function updatePills() {
  const r = (state.meta && state.meta.data_range) || {};
  const db = $("#db-pill");
  db.className = `pill ${r.error ? "bad" : "ok"}`;
  db.replaceChildren();
  db.insertAdjacentHTML("beforeend", '<span class="dot"></span>');
  db.append(r.error ? "Database offline" : `Data to ${fmtDay(r.last).replace(/ \d{4}$/, "")}`);
  db.title = r.error ? `Database not reachable: ${r.error}` : `Records from ${fmtDay(r.first)} to ${fmtDay(r.last)}`;

  const mp = $("#model-pill");
  const ready = aiReady();
  mp.className = `pill pill-link ${ready ? "" : "warn"}`;
  mp.replaceChildren();
  mp.insertAdjacentHTML("beforeend", '<span class="dot"></span>');
  mp.append(ready ? currentModelLabel() : "Connect an AI");
  mp.title = ready ? "AI provider and model. Click to change." : "No AI key yet. Click to add one.";
}

async function loadMeta() {
  try {
    state.meta = await api.meta();
  } catch {
    state.meta = { data_range: { error: "server not reachable" }, providers: {}, defaults: {} };
  }
  updatePills();
}

function route() {
  cancelAllVoice();
  const hash = location.hash.replace(/^#\/?/, "");
  const [name, id] = hash.split("/");
  const view = VIEWS.includes(name) ? name : name === "examples" ? "terms" : "chat";
  document.body.className = `route-${view}`;
  $$(".nav-links a").forEach((a) => {
    if (a.dataset.route === view) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  $$(".views > .view").forEach((v) => { v.hidden = v.dataset.view !== view; });
  const el = $(`#view-${view}`);
  if (view === "chat") showChat(id || null);
  else if (view === "explore") renderExplore(el);
  else if (view === "dictionary") renderDictionary(el);
  else if (view === "terms") renderTerms(el);
  else if (view === "settings") renderSettings(el);
  document.title = { chat: "IPDR Analyst", explore: "Explore · IPDR Analyst", dictionary: "Data dictionary · IPDR Analyst", terms: "Terms · IPDR Analyst", settings: "Settings · IPDR Analyst" }[view];
}

function initSidebar() {
  const btn = $("#sidebar-toggle");
  btn.innerHTML = icon("menu");
  const set = (open) => {
    document.body.classList.toggle("sidebar-open", open);
    btn.setAttribute("aria-expanded", String(open));
  };
  btn.addEventListener("click", () => set(!document.body.classList.contains("sidebar-open")));
  $("#scrim").addEventListener("click", () => set(false));
  $("#conv-list").addEventListener("click", (e) => { if (e.target.closest("a")) set(false); });
}

async function start() {
  initTheme();
  initSidebar();
  initChat();
  onSettingsSaved((quiet) => { updatePills(); refreshVoice(); if (!quiet) toast("Settings saved"); });
  window.addEventListener("hashchange", route);
  route();
  await loadMeta();
  setInterval(async () => {
    try {
      const m = await api.meta();
      const wasDown = state.meta && state.meta.data_range && state.meta.data_range.error;
      state.meta = m;
      updatePills();
      if (wasDown && !m.data_range.error) toast("The database is reachable again.");
    } catch {  }
  }, 120000);
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
else start();

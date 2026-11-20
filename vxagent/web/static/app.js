// The console page. Plain JavaScript, no build step, no dependency.
// Everything the server or a model wrote is put on the page as a text node. Nothing in this
// file parses a string as markup, so no reply can become markup.
"use strict";

const main = document.getElementById("main");
let me = null, page = null, timers = [], events = null, robotAt = 0;

function el(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") n.className = v;
    else if (k === "style") n.style.cssText = v;        // through the style object: the page's policy forbids inline style attributes
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v === true) n.setAttribute(k, "");
    else if (v !== false && v != null) n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    n.appendChild(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return n;
}
const clear = (n) => { while (n.firstChild) n.removeChild(n.firstChild); };
const card = (title, ...kids) => el("section", { class: "card" }, el("h2", {}, title), ...kids);
const tag = (text, kind) => el("span", { class: "tag " + (kind || "") }, text);
const kv = (label, value) => el("div", { class: "kv" }, el("b", {}, label), value == null ? "–" : String(value));
const pre = (obj) => el("pre", {}, typeof obj === "string" ? obj : JSON.stringify(obj, null, 1));
const when = (t) => t ? new Date(t * 1000).toLocaleTimeString() : "–";

async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, { credentials: "same-origin", ...opt });
  let data = null;
  try { data = await r.json(); } catch (e) { data = null; }
  if (r.status === 401 && path !== "/api/login" && path !== "/api/me") { me = null; showLogin(); throw new Error("not logged in"); }
  if (!r.ok) throw new Error((data && data.error) || ("the server answered " + r.status));
  return data;
}

function table(head, rows, onPick) {
  return el("table", {},
    el("thead", {}, el("tr", {}, head.map((h) => el("th", {}, h)))),
    el("tbody", {}, rows.length ? rows.map((r) => el("tr", onPick ? { class: "pick", onclick: () => onPick(r.key) } : {},
      r.cells.map((c) => el("td", {}, c)))) : el("tr", {}, el("td", { colspan: head.length, class: "muted" }, "nothing here"))));
}

function stopTimers() { timers.forEach(clearInterval); timers = []; }
function fail(e) { main.appendChild(el("p", { class: "err" }, String(e.message || e))); }

// ------------------------------------------------------------------ login

async function showLogin() {
  stopTimers();
  if (events) { events.close(); events = null; }
  document.getElementById("top").hidden = true;
  clear(main);
  const about = await fetch("/api/about").then((r) => r.json()).catch(() => ({}));
  const user = el("input", { id: "user", autocomplete: "username" });
  const pass = el("input", { id: "pass", type: "password", autocomplete: "current-password" });
  const msg = el("p", { class: "err" });
  const go = async (ev) => {
    ev.preventDefault();
    try { await api("/api/login", { user_id: user.value.trim(), password: pass.value }); await start(); }
    catch (e) { msg.textContent = e.message; }
  };
  main.appendChild(el("div", { class: "login" },
    card("Log in", el("form", { onsubmit: go }, el("label", {}, "User"), user, el("label", {}, "Password"), pass,
      el("div", { class: "row" }, el("button", { class: "go", type: "submit" }, "Log in")), msg)),
    card("What this is",
      el("p", {}, about.banner || ""),
      el("h3", {}, "Recorded"), el("p", { class: "muted" }, about.recorded || ""),
      el("h3", {}, "Declared or injected"), el("p", { class: "muted" }, about.declared_or_injected || ""),
      el("h3", {}, "Not modelled"), el("p", { class: "muted" }, about.not_modelled || ""),
      el("h3", {}, "Accounts"), el("p", { class: "muted" }, about.accounts || ""))));
}

// ------------------------------------------------------------------ shell

async function start() {
  try { me = await api("/api/me"); } catch (e) { return; }
  document.getElementById("top").hidden = false;
  document.getElementById("whoami").textContent = `${me.display_name || me.user_id} · ${me.role}${me.site ? " · " + me.site : ""}`;
  document.getElementById("stop").hidden = !["operator", "service"].includes(me.role);
  const nav = document.getElementById("nav");
  clear(nav);
  me.pages.forEach((p) => nav.appendChild(el("button", { onclick: () => show(p) }, p)));
  if (["operator", "service", "safety"].includes(me.role)) listen();
  show(me.pages[0]);
}

function show(name) {
  page = name;
  stopTimers();
  clear(main);
  for (const b of document.getElementById("nav").children) b.className = b.textContent === name ? "on" : "";
  const pages = {
    "Robot": robotPage, "Trials": trialsPage, "Run a trial": runTrialPage, "Copilot": chatPage, "Assistant": chatPage,
    "What the copilot can do": copilotPage, "Tickets": ticketsPage, "Diagnosis": ticketsPage, "Calibration": calibrationPage,
    "Propose a change": proposePage, "Change queue": changesPage, "Overridden verdicts": overridesPage,
    "Support grants": grantsPage, "Fleet": fleetPage, "Site health": fleetPage, "Granted tickets": grantedPage,
  };
  (pages[name] || (() => main.appendChild(el("p", {}, "unknown page"))))().catch(fail);
}

function listen() {
  if (events) events.close();
  events = new EventSource("/api/events");           // the browser sends Last-Event-ID when it reconnects
  const again = () => { if (page === "Robot") refreshRobot(); if (page === "Run a trial") refreshApprovals(); };
  ["robot", "approval", "run", "trial", "stop", "site"].forEach((k) => events.addEventListener(k, again));
}

document.getElementById("logout").addEventListener("click", async () => { await api("/api/logout", {}); showLogin(); });
document.getElementById("stop").addEventListener("click", async () => {
  // The stop goes straight to the node. No model is asked.
  const r = await api("/api/stop", {});
  alert(`Stopped ${r.stopped_runs.length} run(s).`);
});

// ------------------------------------------------------------------ operator: robot

let robotBox = null, robotAge = null;
async function robotPage() {
  robotBox = el("div", { class: "grid" });
  robotAge = el("p", { class: "muted" });
  main.appendChild(card("Robot", robotBox, robotAge));
  const poses = await api("/api/robot/poses").catch(() => ({ poses: [] }));
  main.appendChild(card("Named poses", table(["Name", "Offset from start, mm", "Height over pad, mm", "What"],
    poses.poses.map((p) => ({ cells: [p.name, p.offset_from_start_mm.join(", "), p.height_over_pad_mm, p.description] })))));
  if (me.role === "operator") {
    const kind = el("select", {}, ["ecm_moved", "setup_joint_moved", "instrument_changed"].map((k) => el("option", {}, k)));
    main.appendChild(card("Something on the rig was moved", el("p", { class: "muted" }, "Recording this ends the calibration's validity and every approval to move."),
      el("div", { class: "row" }, kind, el("button", { onclick: async () => { pre1.textContent = JSON.stringify(await api("/api/robot/event", { kind: kind.value }), null, 1); } }, "Record")),
      pre1));
  }
  await refreshRobot();
  timers.push(setInterval(refreshRobot, 2000));
  timers.push(setInterval(() => {
    const age = (Date.now() - robotAt) / 1000;
    robotAge.textContent = `state read ${age.toFixed(0)} s ago`;
    robotBox.className = "grid" + (age > 5 ? " stale" : "");     // a state that is old is shown as old
  }, 500));
}
const pre1 = el("pre", {});
async function refreshRobot() {
  if (!robotBox) return;
  let s;
  try { s = await api("/api/robot"); } catch (e) { return; }
  robotAt = Date.now();
  const m = s.measured || {};
  clear(robotBox);
  robotBox.append(kv("Site / arm", `${s.site} / ${s.arm}`), kv("Backend", s.backend), kv("Powered", s.powered), kv("Enabled", s.enabled),
    kv("Homed", s.homed), kv("Busy", s.busy), kv("Fault", s.fault || "none"), kv("Tip position, mm", (m.position_mm || []).join(", ")),
    kv("Height over pad, mm", m.height_over_pad_mm), kv("Jaw, deg", s.jaw_deg), kv("Safety profile", s.safety_profile),
    kv("State number", s.state_seq), kv("Needle (declared)", s.needle ? (s.needle.held ? "held" : "not held") : "–"));
}

// ------------------------------------------------------------------ operator: trials

async function trialsPage() {
  const list = await api("/api/trials");
  const detail = el("div", {});
  main.appendChild(card("Trials at " + list.site, table(["Trial", "Status", "Outcome", "Machine verdict", "Operator"],
    list.trials.map((t) => ({ key: t.run_id, cells: [t.run_id, t.waiting_for ? "waiting: " + t.waiting_for : t.status, t.outcome || "", t.machine_verdict || "",
      t.operator_decision ? `${t.operator_decision}${t.operator_verdict ? " → " + t.operator_verdict : ""}` : ""] })), (id) => showTrial(id, detail))));
  main.appendChild(detail);
}

async function showTrial(id, into) {
  const t = await api("/api/trials/" + id);
  clear(into);
  const picked = el("pre", {}, "select a mark on the time axis");
  const line = el("div", { class: "timeline" });
  const ts = t.timeline.map((e) => e.at).filter(Boolean);
  const t0 = Math.min(...ts), span = Math.max(1e-6, Math.max(...ts) - t0);
  t.timeline.forEach((e) => {
    const cls = e.what.startsWith("approval") ? "approval" : e.what === "model_call" ? "model"
      : /fail|reject|stop|watchdog|refused/.test(e.what) || (e.data && e.data.status && e.data.status !== "succeeded") ? "bad" : "";
    const mark = el("i", { class: cls, style: `left:${(((e.at - t0) / span) * 99).toFixed(2)}%`, title: e.what });
    mark.addEventListener("click", () => {
      for (const m of line.children) m.classList.remove("sel");
      mark.classList.add("sel");
      const pose = (t.poses || []).filter((p) => p.at <= e.at).pop();       // the twin's pose as last observed before this moment
      picked.textContent = JSON.stringify({ at: when(e.at), event: e.what, ...e.data, robot_state_number: e.robot_seq,
        twin_pose_then: pose ? { position_mm: pose.position_mm, height_over_pad_mm: pose.height_over_pad_mm, jaw_deg: pose.jaw_deg } : "not observed yet" }, null, 1);
    });
    line.appendChild(mark);
  });
  const buttons = el("div", { class: "row" });
  if (me.role === "operator" && t.waiting_for && t.waiting_for.kind === "approval") {
    buttons.append(el("button", { class: "go", onclick: () => act(`/api/trials/${id}/go`, { approve: true, wait_s: 120 }) }, "Go"),
      el("button", { class: "no", onclick: () => act(`/api/trials/${id}/go`, { approve: false, wait_s: 10 }) }, "Do not run"));
  }
  if (me.role === "operator" && t.waiting_for && t.waiting_for.kind === "operator_confirmation") {
    const reason = el("input", { placeholder: "reason, needed for an override" });
    const to = el("select", {}, ["failed", "success"].map((k) => el("option", {}, k)));
    buttons.append(el("button", { class: "go", onclick: () => act(`/api/trials/${id}/verdict`, { decision: "confirm", wait_s: 30 }) }, "Confirm the machine verdict"),
      el("span", { class: "grow" }, reason), to,
      el("button", { class: "no", onclick: () => act(`/api/trials/${id}/verdict`, { decision: "override", to: to.value, reason: reason.value, wait_s: 30 }) }, "Override"));
  }
  async function act(path, body) { try { await api(path, body); } catch (e) { alert(e.message); } showTrial(id, into); }
  into.appendChild(card("Trial " + id,
    el("div", { class: "grid" }, kv("Status", t.waiting_for ? "waiting: " + t.waiting_for.kind : t.status), kv("Outcome", t.outcome),
      kv("Machine verdict", t.verdict ? t.verdict.machine : null), kv("Operator", t.operator_verdict ? t.operator_verdict.decision : null),
      kv("Needle", t.needle), kv("Stages", (t.stages_run || []).join(" → "))),
    buttons,
    t.repairs ? el("p", {}, "Repair by: ", t.repairs.join(", ")) : null,
    t.summary ? el("div", { class: "answer" }, t.summary) : null,
    t.summary ? el("p", { class: "muted" }, "summary written by: " + t.summary_source) : null,
    el("h3", {}, "Time axis: stages, approvals (green), model calls (violet), rejections and stops (red)"), line, picked,
    el("details", {}, el("summary", {}, "pre-flight, rehearsal and verdict as recorded"), pre({ preflight: t.preflight, rehearsal: t.rehearsal, verdict: t.verdict, operator: t.operator_verdict }))));
}

let approvalBox = null;
async function runTrialPage() {
  const out = el("div", {});
  const needle = el("select", {}, [["holds", "{}"], ["slips on the lift", '{"slips_on_lift":true}'], ["slips while held", '{"slips_while_holding":true}'],
    ["the grasp misses", '{"grasp_succeeds":false}']].map(([t, v]) => el("option", { value: v }, t)));
  const ticket = el("input", { type: "checkbox", style: "width:auto" });
  main.appendChild(card("Run a trial",
    el("p", { class: "muted" }, "Pre-flight, rehearsal on a copy of the twin, your go, reach along a recorded expert rollout, grasp, retention, place, verdict. The needle is declared, not observed."),
    el("label", {}, "Declared behaviour of the needle in this run"), needle,
    el("div", { class: "row" }, ticket, el("span", {}, "if pre-flight fails, open a ticket for the service engineer")),
    el("div", { class: "row" }, el("button", { class: "go", onclick: async () => {
      try { const t = await api("/api/trials", { needle: JSON.parse(needle.value), open_ticket: ticket.checked }); await showTrial(t.run_id, out); }
      catch (e) { clear(out); out.appendChild(el("p", { class: "err" }, e.message)); }
    } }, "Start"))));
  approvalBox = el("div", {});
  main.appendChild(card("Approval cards", el("p", { class: "muted" }, "Motions the copilot or the service engineer asked for. A card is bound to the robot state it shows."), approvalBox));
  main.appendChild(out);
  await refreshApprovals();
  timers.push(setInterval(refreshApprovals, 3000));
}
async function refreshApprovals() {
  if (!approvalBox || me.role !== "operator") return;
  let list;
  try { list = await api("/api/approvals"); } catch (e) { return; }
  clear(approvalBox);
  if (!list.pending.length) approvalBox.appendChild(el("p", { class: "muted" }, "none waiting"));
  list.pending.filter((c) => c.tool !== "operator_go").forEach((c) => approvalBox.appendChild(approvalCard(c, refreshApprovals)));
}
function approvalCard(c, after) {
  const decide = async (approve) => { try { await api("/api/approvals/" + c.approval_id, { approve }); } catch (e) { alert(e.message); } after(); };
  return el("div", { class: "card" },
    el("div", { class: "grid" }, kv("Asked by", c.requested_by), kv("Tool", c.tool), kv("What moves", c.what_moves), kv("Profile in force", c.profile),
      kv("From (tip, mm)", c.from && c.from.position_mm ? c.from.position_mm.join(", ") : null),
      kv("To", c.to ? (c.to.named_pose ? `${c.to.named_pose}, ${c.to.distance_mm} mm away, ${c.to.height_over_pad_mm} mm over the pad` : JSON.stringify(c.to)) : null),
      kv("Rehearsal on the twin", c.rehearsal ? (c.rehearsal.would_complete ? `completes in ${c.rehearsal.control_steps || "?"} steps` : "does not complete") : "none"),
      kv("Bound to state number", c.bound_to.robot_state_seq), kv("Valid for", `${Math.round(c.seconds_left)} s`)),
    el("div", { class: "row" }, el("button", { class: "go", onclick: () => decide(true) }, "Approve"), el("button", { class: "no", onclick: () => decide(false) }, "Reject")));
}

// ------------------------------------------------------------------ copilot, for every role

async function chatPage() {
  const log = el("div", {});
  const box = el("textarea", { placeholder: "Ask the copilot" });
  const send = el("button", { class: "go" }, "Send");
  const render = (out) => {
    log.appendChild(el("div", { class: "answer" }, out.answer || (out.status === "awaiting_approval" ? "Waiting for the operator's approval." : "(no answer)")));
    if (out.approval) log.appendChild(me.role === "operator" ? approvalCard(out.approval, async () => {}) : el("p", { class: "muted" }, "An operator of this site has to approve the motion on their screen."));
    const t = out.trace;
    log.appendChild(el("details", {}, el("summary", {}, `trace: ${t.model_calls} model calls, ${t.tool_calls} tool calls, ${t.tokens} tokens, ${Math.round(t.model_ms)} ms in the model, stopped because: ${out.stop_reason}`),
      table(["Step", "Name", "ok", "ms", "Input", "Output"], t.steps.map((s) => ({ cells: [s.kind, s.name, s.ok ? "yes" : (s.error_kind || "no"), s.latency_ms,
        s.input ? JSON.stringify(s.input) : (s.tokens ? `${s.tokens.prompt}+${s.tokens.completion} tokens` : ""), s.output ? JSON.stringify(s.output).slice(0, 600) : ""] })))));
  };
  send.addEventListener("click", async () => {
    const text = box.value.trim();
    if (!text) return;
    log.appendChild(el("p", {}, el("b", {}, "You: "), text));
    box.value = ""; send.disabled = true;
    try { render(await api("/api/chat", { message: text })); } catch (e) { log.appendChild(el("p", { class: "err" }, e.message)); }
    send.disabled = false;
  });
  main.appendChild(card(me.role === "support" ? "Assistant" : "Copilot", el("p", { class: "muted" }, "Answers come from tools, with their trace. A motion needs the operator's approval."), log, box, el("div", { class: "row" }, send)));
}

async function copilotPage() {
  const c = await api("/api/copilot");
  main.appendChild(card("What the copilot can do", el("p", {}, "It cannot: " + c.cannot),
    table(["Tool", "Tier", "Approval", "What it does"], c.tools.map((t) => ({ cells: [t.name, t.tier, t.needs_approval ? "operator" : "", t.what] })))));
}

// ------------------------------------------------------------------ service engineer

async function ticketsPage() {
  const list = await api("/api/tickets");
  const detail = el("div", {});
  main.appendChild(card("Tickets at " + list.site, table(["#", "Symptom", "Source", "Status", "Finding"],
    list.tickets.map((t) => ({ key: t.ticket_id, cells: [t.ticket_id, t.symptom, t.source, t.status, t.finding || ""] })), (id) => showTicket(id, detail))));
  main.appendChild(detail);
}
async function showTicket(id, into) {
  const t = await api("/api/tickets/" + id);
  clear(into);
  const out = el("div", {});
  const run = async (path, body) => { out.textContent = "working…"; try { const r = await api(path, body); clear(out); out.appendChild(pre(r)); } catch (e) { out.textContent = e.message; } };
  const buttons = me.role === "service" && t.status === "open" ? el("div", { class: "row" },
    el("button", { onclick: () => run(`/api/tickets/${id}/diagnose`, { method: "eliminate" }) }, "Diagnose by elimination"),
    el("button", { onclick: () => run(`/api/tickets/${id}/diagnose`, { method: "team" }) }, "Diagnose with the triage team"),
    el("button", { onclick: () => run(`/api/tickets/${id}/close`, {}) }, "Close with a service report")) : null;
  into.appendChild(card(`Ticket ${id}: ${t.status}`, el("p", {}, t.symptom), buttons, out,
    table(["Record", "Kind", "By", "Data"], t.records.map((r) => ({ cells: [r.record, r.kind, r.by, JSON.stringify(r.data).slice(0, 700)] })))));
}

async function calibrationPage() {
  const c = await api("/api/calibration");
  main.appendChild(card("Calibration", el("p", { class: "muted" }, c.camera),
    el("div", { class: "grid" }, kv("Record in use", c.in_use ? c.in_use.id : "none"), kv("Valid", c.validity.valid), kv("Why", c.validity.why),
      kv("Draft", c.draft ? `${c.draft.id} (${c.draft.status})` : "none")),
    c.draft ? pre(c.draft.held_out_residual) : null,
    el("p", { class: "muted" }, "A new calibration is collected with the runbook collect_calibration_samples (ask the copilot); the operator approves the motion. The fit is a draft until the safety lead approves its activation.")));
  if (me.role === "service" && c.draft && c.draft.status === "draft") {
    const reason = el("input", { placeholder: "reason" });
    const out = el("pre", {});
    main.appendChild(card("Propose the draft", reason, el("div", { class: "row" }, el("button", { class: "go", onclick: async () => {
      try { out.textContent = JSON.stringify(await api("/api/changes", { kind: "calibration", reason: reason.value }), null, 1); } catch (e) { out.textContent = e.message; }
    } }, "Send to the safety lead")), out));
  }
}

async function proposePage() {
  const cfg = await api("/api/config");
  const path = el("select", {}, cfg.paths_a_change_may_touch.map((p) => el("option", { value: p.path }, `${p.path}  (now ${JSON.stringify(p.now)} ${p.file_stores || ""})`)));
  const value = el("input", { placeholder: "new value" }), unit = el("input", { placeholder: "unit, e.g. mm", value: "mm" });
  const limit = el("select", {}, ["none", "tighten", "loosen"].map((k) => el("option", {}, k)));
  const reason = el("input", { placeholder: "why" }), evidence = el("input", { placeholder: "record ids, comma separated (e.g. ticket:1/evidence/1)" });
  const out = el("pre", {});
  main.appendChild(card("Propose a change to " + cfg.target, el("p", { class: "muted" }, `version ${cfg.version}. A proposal changes nothing; the safety lead decides.`),
    el("label", {}, "Path"), path, el("label", {}, "Value and unit"), el("div", { class: "row" }, value, unit),
    el("label", {}, "Effect on a safety limit"), limit, el("label", {}, "Reason"), reason, el("label", {}, "Evidence"), evidence,
    el("div", { class: "row" }, el("button", { class: "go", onclick: async () => {
      const v = value.value.trim();
      try {
        out.textContent = JSON.stringify(await api("/api/changes", { target: cfg.target, limit_change: limit.value, reason: reason.value,
          evidence: evidence.value.split(",").map((s) => s.trim()).filter(Boolean),
          operations: [{ op: "replace", path: path.value, value: isNaN(Number(v)) ? v : Number(v), unit: unit.value.trim() || "none" }] }), null, 1);
      } catch (e) { out.textContent = e.message; }
    } }, "File the proposal")), out));
}

// ------------------------------------------------------------------ safety lead

async function changesPage() {
  const q = await api("/api/changes");
  if (!q.changes.length) main.appendChild(card("Change queue", el("p", { class: "muted" }, "no proposal is waiting")));
  q.changes.forEach((c) => {
    const reason = el("input", { placeholder: "reason (required if a limit is loosened)" });
    const out = el("pre", {});
    const decide = async (approve) => { try { out.textContent = JSON.stringify(await api(`/api/changes/${c.proposal_id}/decision`, { approve, reason: reason.value }), null, 1); } catch (e) { out.textContent = e.message; } };
    main.appendChild(card(`Proposal ${c.proposal_id}: ${c.target}`,
      c.loosens_a_limit ? tag("loosens a safety limit", "bad") : tag(c.limit_effect === "tighten" ? "tightens a limit" : "no limit touched", "ok"),
      el("div", { class: "grid" }, kv("Proposed by", c.proposer), kv("Kind", c.kind), kv("Validation", c.validation), kv("Evidence", (c.evidence || []).join(", "))),
      el("p", {}, c.reason), el("h3", {}, "Operations"), pre(c.operations), el("h3", {}, "Effect on the twin"), pre(c.effect_on_the_twin),
      me.role === "safety" ? el("div", { class: "row" }, el("span", { class: "grow" }, reason), el("button", { class: "go", onclick: () => decide(true) }, "Approve and apply"),
        el("button", { class: "no", onclick: () => decide(false) }, "Reject")) : null, out));
  });
}
async function overridesPage() {
  const o = await api("/api/overrides");
  main.appendChild(card("Verdicts an operator overrode", table(["Trial", "Operator", "Machine verdict", "Operator's verdict", "Reason"],
    o.overridden_verdicts.map((t) => ({ cells: [t.run_id, t.operator, t.machine_verdict, t.operator_verdict, t.reason] })))));
}
async function grantsPage() {
  const g = await api("/api/grants");
  const ticket = el("input", { placeholder: "ticket number" }), who = el("input", { placeholder: "support engineer", value: "sup.wong" }), hours = el("input", { placeholder: "hours", value: "4" });
  const out = el("pre", {});
  main.appendChild(card("Support grants at " + g.site, table(["#", "Ticket", "To", "By", "Ends", "State", ""],
    g.grants.map((x) => ({ cells: [x.grant_id, x.ticket_id, x.grantee, x.granted_by, new Date(x.expires * 1000).toLocaleString(), x.state,
      x.state === "active" ? el("button", { class: "no", onclick: async () => { await api(`/api/grants/${x.grant_id}/revoke`, {}); show("Support grants"); } }, "Revoke") : ""] })))));
  main.appendChild(card("Give read access to one ticket", el("div", { class: "row" }, ticket, who, hours, el("button", { class: "go", onclick: async () => {
    try { out.textContent = JSON.stringify(await api("/api/grants", { ticket_id: Number(ticket.value), grantee: who.value.trim(), hours: Number(hours.value) }), null, 1); } catch (e) { out.textContent = e.message; }
  } }, "Grant")), out));
}

// ------------------------------------------------------------------ remote support

async function fleetPage() {
  const o = await api("/api/fleet/overview");
  const detail = el("div", {});
  main.appendChild(card("Fleet", el("p", { class: "muted" }, "Counts, kinds and timings only. Nothing of a site's work is here."),
    table(["Site", "In contact", "Healthy", "Software", "Config", "Runs", "Trials"], o.sites.map((s) => ({ key: s.site, cells: [s.site,
      s.in_contact ? tag("yes", "ok") : tag(`no, ${Math.round(s.last_contact_s_ago)} s`, "bad"), s.healthy ? tag("yes", "ok") : tag("no", "warn"), s.software, s.config_version, s.runs, s.trials] })),
    async (site) => { clear(detail); detail.appendChild(card("Site " + site, pre(await api("/api/fleet/sites/" + site)))); })));
  const d = await api("/api/fleet/drift");
  main.appendChild(card("Version drift", d.drift ? pre(d.differs) : el("p", { class: "muted" }, "all sites report the same versions")));
  main.appendChild(detail);
}
async function grantedPage() {
  const site = el("input", { placeholder: "site, e.g. site-a" }), ticket = el("input", { placeholder: "ticket number" });
  const out = el("pre", {});
  main.appendChild(card("Read a granted ticket", el("p", { class: "muted" }, "Needs a grant from that site's safety lead. The site checks it and records the read."),
    el("div", { class: "row" }, site, ticket, el("button", { onclick: async () => {
      try { out.textContent = JSON.stringify(await api(`/api/fleet/tickets/${encodeURIComponent(site.value.trim())}/${Number(ticket.value)}`), null, 1); } catch (e) { out.textContent = e.message; }
    } }, "Read")), out));
}

start().then(() => { if (!me) showLogin(); });

"""
build_site.py
=============
Generate a self-contained, offline HTML dashboard of the current Elo rankings.

    python build_site.py

Reads the master results file, runs the tuned "production" Elo, and writes:
    site/index.html    -- double-clickable dashboard (all data embedded, no deps)
    site/rankings.json -- the same data as JSON, for programmatic use

The data is embedded directly in the HTML, so it works from file:// with no web
server. Re-run this after update_data.py and the dashboard reflects new matches.
"""
import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

import rugby_elo as re

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, "data", "rugby_results_master.csv")
HIST = os.path.join(HERE, "data", "rugby_results.csv")
SITE = os.path.join(HERE, "site")
os.makedirs(SITE, exist_ok=True)

# Tuned "production" configuration (see run_baseline.py; stable interior optimum)
PROD = re.EloConfig(k=43.0, hfa=110.0)

# Per-team display metadata: abbreviation + line/badge colour.
TEAM_META = {
    "South Africa": {"abbr": "RSA", "color": "#047857"},
    "New Zealand":  {"abbr": "NZL", "color": "#111827"},
    "Ireland":      {"abbr": "IRE", "color": "#22c55e"},
    "France":       {"abbr": "FRA", "color": "#2563eb"},
    "England":      {"abbr": "ENG", "color": "#dc2626"},
    "Argentina":    {"abbr": "ARG", "color": "#38bdf8"},
    "Scotland":     {"abbr": "SCO", "color": "#1e3a8a"},
    "Australia":    {"abbr": "AUS", "color": "#f59e0b"},
    "Wales":        {"abbr": "WAL", "color": "#9d174d"},
    "Italy":        {"abbr": "ITA", "color": "#0ea5e9"},
}


def build_payload():
    data_path = MASTER if os.path.exists(MASTER) else HIST
    df = re.load_results(data_path, tier1_only=True)
    pred, model, hist = re.run_elo(df, PROD, track_history=True)

    last_date = df["date"].max()

    # --- daily rating snapshots (one per date), forward-filled --------------
    h = hist.drop_duplicates("date", keep="last").set_index("date").sort_index()
    h = h.ffill()
    cur = h.iloc[-1]
    target = last_date - timedelta(days=365)
    prior_rows = h.loc[:target]
    prior = prior_rows.iloc[-1] if len(prior_rows) else cur

    # --- annual series for the chart ---------------------------------------
    annual = h.resample("YE").last().ffill()
    years = [int(y) for y in annual.index.year]
    series = {}
    for t in TEAM_META:
        if t in annual.columns:
            series[t] = [None if pd.isna(v) else round(float(v), 1)
                         for v in annual[t]]

    # --- ranking rows with movement + form ---------------------------------
    rank_df = re.current_ranking(model)
    ranking = []
    for i, row in rank_df.iterrows():
        team = row["team"]
        delta = float(cur[team] - prior[team]) if team in prior else 0.0
        ranking.append({
            "rank": i + 1,
            "team": team,
            "abbr": TEAM_META[team]["abbr"],
            "color": TEAM_META[team]["color"],
            "rating": round(float(row["rating"]), 1),
            "games": int(row["games"]),
            "delta": round(delta, 1),
            "form": _recent_form(df, team, n=5),
        })

    # --- recent results (tier-1) -------------------------------------------
    recent = []
    for _, r in df.tail(14).iloc[::-1].iterrows():
        recent.append({
            "date": r["date"].strftime("%Y-%m-%d"),
            "home": r["home_team"], "hs": int(r["home_score"]),
            "away": r["away_team"], "as": int(r["away_score"]),
            "comp": str(r["competition"]), "neutral": bool(r["neutral"]),
            "wc": bool(r["world_cup"]),
        })

    # --- headline predictive metrics (prequential, 2000+) ------------------
    m = re.evaluate(pred, start_year=2000)
    model = _fit_match_models(pred, PROD.hfa)

    return {
        "generated": date.today().strftime("%Y-%m-%d"),
        "data_through": last_date.strftime("%Y-%m-%d"),
        "n_matches": int(len(df)),
        "config": {"k": PROD.k, "hfa": PROD.hfa},
        "model": model,
        "metrics": {"accuracy": round(m["accuracy"], 3),
                    "log_loss": round(m["log_loss"], 3),
                    "brier": round(m["brier"], 3), "n": m["n"]},
        "ranking": ranking,
        "history": {"years": years, "series": series},
        "recent": recent,
    }


def _fit_match_models(pred, hfa, since_year=1970):
    """Fit two auxiliary models the client calculator needs, from the same
    prequential predictions:

      * draw probability   p_draw(d) = a * exp(-b*|d|)   (d = effective Elo gap)
      * expected margin     margin   = slope * d          (through the origin)

    These convert Elo's expected score into a win / draw / loss split and a
    predicted points margin. Fitted on the modern era (since_year+).
    """
    p = pred[pred["date"].dt.year >= since_year]
    eff = (p["rating_home_pre"] + np.where(p["neutral"], 0.0, hfa)
           - p["rating_away_pre"]).to_numpy()
    margin = p["margin"].to_numpy()
    is_draw = (p["s_home"].to_numpy() == 0.5).astype(float)
    d = np.abs(eff)

    slope = float((eff * margin).sum() / (eff * eff).sum())

    a, b = 0.09, 0.004     # sensible fallback
    try:
        from scipy.optimize import curve_fit
        edges = np.arange(0, d.max() + 40, 40)
        idx = np.digitize(d, edges)
        xs, ys, ws = [], [], []
        for bi in range(1, len(edges) + 1):
            mask = idx == bi
            if mask.sum() >= 25:
                xs.append(d[mask].mean())
                ys.append(is_draw[mask].mean())
                ws.append(mask.sum())
        popt, _ = curve_fit(lambda x, a, b: a * np.exp(-b * x),
                            np.array(xs), np.array(ys),
                            p0=[0.09, 0.004], sigma=1/np.sqrt(np.array(ws)),
                            bounds=([0, 0], [0.5, 0.05]), maxfev=20000)
        a, b = float(popt[0]), float(popt[1])
    except Exception as e:
        print(f"  [draw-model fallback: {e}]")

    return {"scale": 400.0, "hfa": float(hfa),
            "draw_a": round(a, 5), "draw_b": round(b, 6),
            "margin_slope": round(slope, 5)}


def _recent_form(df, team, n=5):
    sub = df[(df["home_team"] == team) | (df["away_team"] == team)].tail(n)
    out = []
    for _, r in sub.iterrows():
        home = r["home_team"] == team
        gf = r["home_score"] if home else r["away_score"]
        ga = r["away_score"] if home else r["home_score"]
        res = "W" if gf > ga else ("L" if gf < ga else "D")
        opp = r["away_team"] if home else r["home_team"]
        venue = "N" if r["neutral"] else ("H" if home else "A")
        out.append({"r": res, "opp": opp, "gf": int(gf), "ga": int(ga),
                    "venue": venue, "date": r["date"].strftime("%Y-%m-%d")})
    return out


def main():
    payload = build_payload()
    with open(os.path.join(SITE, "rankings.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    html = TEMPLATE.replace("__DATA_JSON__", json.dumps(payload))
    out = os.path.join(SITE, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    # body-only fragment (no doctype/html/head/body) for inline preview widgets
    frag = html[html.index("<!--WIDGET-START-->"):html.index("<!--WIDGET-END-->")]
    with open(os.path.join(SITE, "_widget.html"), "w", encoding="utf-8") as f:
        f.write(frag)

    top = payload["ranking"][0]
    print(f"Wrote {out}")
    print(f"  data through {payload['data_through']} | {payload['n_matches']} matches")
    print(f"  #1 {top['team']} ({top['rating']})  | accuracy {payload['metrics']['accuracy']}")


# --------------------------------------------------------------------------- #
# HTML/CSS/JS template. Data is injected at __DATA_JSON__. No external deps.
# --------------------------------------------------------------------------- #
TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rugby Elo Rankings</title>
</head>
<body>
<!--WIDGET-START-->
<style>
  :root{
    --bg:#eef1f5; --card:#ffffff; --ink:#0f172a; --muted:#64748b;
    --line:#e5e9f0; --win:#16a34a; --draw:#94a3b8; --loss:#dc2626;
    --accent:#0b1f3a;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       -webkit-font-smoothing:antialiased}
  header{background:linear-gradient(135deg,#0b1f3a,#123a63);color:#fff;
         padding:26px 20px}
  header .wrap{max-width:1080px;margin:0 auto}
  header h1{margin:0;font-size:24px;letter-spacing:.2px}
  header p{margin:6px 0 0;color:#b9c6da;font-size:13px}
  .wrap{max-width:1080px;margin:0 auto;padding:20px}
  .grid{display:grid;grid-template-columns:1fr;gap:18px}
  @media(min-width:880px){.grid{grid-template-columns:1.35fr .9fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;
        box-shadow:0 1px 3px rgba(15,23,42,.05);padding:16px 18px}
  .card h2{margin:0 0 12px;font-size:14px;text-transform:uppercase;
           letter-spacing:.6px;color:var(--muted)}
  table{width:100%;border-collapse:collapse}
  th,td{text-align:left;padding:9px 8px;font-size:14px;border-bottom:1px solid var(--line)}
  th{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted)}
  tbody tr:last-child td{border-bottom:none}
  tr:hover td{background:#f8fafc}
  .rk{width:34px;color:var(--muted);font-variant-numeric:tabular-nums;font-weight:600}
  .medal{font-size:15px}
  .badge{display:inline-block;min-width:40px;text-align:center;color:#fff;
         font-weight:700;font-size:11px;letter-spacing:.5px;border-radius:6px;
         padding:3px 7px;margin-right:9px;vertical-align:middle}
  .team{font-weight:600;vertical-align:middle}
  .rating{font-weight:700;font-variant-numeric:tabular-nums}
  .delta{font-variant-numeric:tabular-nums;font-weight:600;font-size:13px}
  .up{color:var(--win)} .down{color:var(--loss)} .flat{color:var(--muted)}
  .form{white-space:nowrap}
  .dot{display:inline-block;width:15px;height:15px;border-radius:50%;
       margin-right:3px;font-size:9px;line-height:15px;text-align:center;
       color:#fff;font-weight:700;cursor:default}
  .games{color:var(--muted);font-variant-numeric:tabular-nums}
  .calc{margin-bottom:18px}
  .calcrow{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  .calcrow select{flex:1;min-width:150px;padding:10px 11px;border:1px solid var(--line);
    border-radius:9px;font-size:15px;font-weight:600;background:#fff;color:var(--ink)}
  .vs{color:var(--muted);font-weight:700}
  .venue{display:flex;gap:6px;margin:12px 0 16px}
  .venue button{flex:1;border:1px solid var(--line);background:#fff;border-radius:9px;
    padding:8px;font-size:13px;cursor:pointer;color:var(--muted);font-weight:600}
  .venue button.active{background:var(--accent);color:#fff;border-color:var(--accent)}
  .probbar{display:flex;height:38px;border-radius:9px;overflow:hidden;border:1px solid var(--line)}
  .probbar div{display:flex;align-items:center;justify-content:center;color:#fff;
    font-weight:700;font-size:13px;transition:width .25s;white-space:nowrap;overflow:hidden}
  .probleg{display:flex;justify-content:space-between;margin-top:9px;font-size:13px;color:var(--muted)}
  .probleg b{color:var(--ink)}
  .pred{margin-top:12px;font-size:15px;font-weight:700;text-align:center;color:var(--accent)}
  .about .lead{font-size:15px;color:var(--ink);margin:0 0 4px}
  .about h3{font-size:12px;text-transform:uppercase;letter-spacing:.5px;color:var(--accent);
    margin:18px 0 6px}
  .about p,.about li{font-size:14px;line-height:1.65;color:#334155;max-width:74ch}
  .about ul{margin:6px 0 0;padding-left:20px}
  .about li{margin-bottom:7px}
  .about li b{color:var(--ink)}
  .about .note{font-size:12.5px;color:var(--muted);font-style:italic;margin-top:16px}
  .controls{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:8px}
  .rangebtn{border:1px solid var(--line);background:#fff;border-radius:8px;
            padding:4px 10px;font-size:12px;cursor:pointer;color:var(--muted)}
  .rangebtn.active{background:var(--accent);color:#fff;border-color:var(--accent)}
  .legend{display:flex;flex-wrap:wrap;gap:5px;margin-top:10px}
  .chip{display:flex;align-items:center;gap:5px;border:1px solid var(--line);
        border-radius:20px;padding:2px 9px 2px 6px;font-size:11.5px;cursor:pointer;
        user-select:none}
  .chip .sw{width:10px;height:10px;border-radius:50%}
  .chip.off{opacity:.35;text-decoration:line-through}
  .chartbox{position:relative}
  svg{width:100%;display:block}
  .grid line{stroke:#eef1f5}
  .ylab{fill:var(--muted);font-size:10px;text-anchor:end}
  .xlab{fill:var(--muted);font-size:10px;text-anchor:middle}
  .tip{position:absolute;pointer-events:none;background:#0b1f3aee;color:#fff;
       border-radius:8px;padding:8px 10px;font-size:12px;min-width:150px;
       transform:translate(-9999px,-9999px);box-shadow:0 4px 14px rgba(0,0,0,.25)}
  .tip .ty{font-weight:700;margin-bottom:4px}
  .tip .row{display:flex;justify-content:space-between;gap:14px}
  .tip .row .sw{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
  .vline{stroke:#94a3b8;stroke-dasharray:3 3}
  .rec{list-style:none;margin:0;padding:0}
  .rec li{display:flex;align-items:center;gap:8px;padding:7px 2px;
          border-bottom:1px solid var(--line);font-size:13px}
  .rec li:last-child{border-bottom:none}
  .rec .d{color:var(--muted);font-size:11px;width:74px;flex:none;font-variant-numeric:tabular-nums}
  .rec .mu{flex:1}
  .rec .sc{font-weight:700;font-variant-numeric:tabular-nums}
  .rec .tag{font-size:10px;color:var(--muted);border:1px solid var(--line);
            border-radius:5px;padding:1px 5px;flex:none}
  .win{font-weight:700}
  footer{max-width:1080px;margin:0 auto;padding:6px 20px 34px;color:var(--muted);
         font-size:12px;line-height:1.6}
  code{background:#e2e8f0;border-radius:4px;padding:1px 5px;font-size:11.5px}
</style>
<header><div class="wrap">
  <h1>🏉 International Rugby Elo Rankings</h1>
  <p id="sub"></p>
</div></header>

<div class="wrap">
  <div class="card calc">
    <h2>Win probability calculator</h2>
    <div class="calcrow">
      <select id="teamA"></select>
      <div class="vs">vs</div>
      <select id="teamB"></select>
    </div>
    <div class="venue" id="venue">
      <button data-v="A">A at home</button>
      <button data-v="N" class="active">Neutral</button>
      <button data-v="B">B at home</button>
    </div>
    <div class="probbar" id="probbar"></div>
    <div class="probleg" id="probleg"></div>
    <div class="pred" id="pred"></div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Rankings</h2>
      <table><thead><tr>
        <th class="rk">#</th><th>Team</th><th>Rating</th>
        <th title="Change over the last 12 months">1yr</th>
        <th>Form</th><th>P</th>
      </tr></thead><tbody id="tbody"></tbody></table>
    </div>
    <div class="card">
      <h2>Recent tier-1 results</h2>
      <ul class="rec" id="recent"></ul>
    </div>
  </div>

  <div class="card" style="margin-top:18px">
    <h2>Rating history</h2>
    <div class="controls" id="ranges"></div>
    <div class="chartbox">
      <svg id="chart" height="420"></svg>
      <div class="tip" id="tip"></div>
    </div>
    <div class="legend" id="legend"></div>
  </div>

  <div class="card about" style="margin-top:18px">
    <h2>About &mdash; how it works</h2>
    <p class="lead">A live Elo rating of the 10 tier-1 rugby nations, updated automatically after every Test match.</p>

    <h3>What you're looking at</h3>
    <p>The table ranks the teams by rating, shows whether each is rising or sliding over the
    last year, and their last five results. The chart tracks how every nation's strength has
    risen and fallen over the decades, and the calculator at the top lets you pick any two
    teams and a venue to see the win, draw and loss chances plus a predicted margin.</p>

    <h3>How the ranking works</h3>
    <p>Every team carries a points score. When two teams play, points move from the loser to
    the winner &mdash; a bit like a bet settling. How many move depends on the upset: beat a
    side you were expected to beat and you gain a little; topple a much stronger team and you
    gain a lot (and they lose a lot). Playing at home earns a boost, because home advantage is
    worth real points in rugby. The table updates after every match, so it reflects current
    form rather than reputation. It's the same idea used to rank chess players and in the FIFA
    world rankings.</p>

    <h3>How it differs from the official World Rugby ranking</h3>
    <ul>
      <li><b>Tuned to be accurate.</b> We tested it against thousands of past matches and set
      it up to predict results as well as possible &mdash; it calls about <b>3 in 4</b> games
      correctly. World Rugby's settings are fixed by committee, not optimised.</li>
      <li><b>No artificial caps.</b> World Rugby limits how much a single result can move the
      table, which makes it slow to react. Ours lets a genuine shock count for what it's worth,
      so slumps and surges show up clearly and quickly.</li>
      <li><b>The full story.</b> It uses every international back to 1871, not just a rolling
      recent snapshot.</li>
      <li><b>Open and interactive.</b> The code and data are public, and you can test any
      matchup yourself instead of just being handed a number.</li>
    </ul>

    <p class="note">A passion project, not an official ranking &mdash; but the maths is honest,
    and every bit of it is open to check.</p>
  </div>
</div>

<footer id="foot"></footer>

<script>
const DATA = __DATA_JSON__;
const META = {}; DATA.ranking.forEach(r => META[r.team] = r);
const TEAMS = DATA.ranking.map(r => r.team);
const hidden = new Set();
let range = 2000;

// ---- header + footer ----
document.getElementById('sub').textContent =
  `Updated ${DATA.generated} · data through ${DATA.data_through} · ${DATA.n_matches.toLocaleString()} tier-1 tests since 1871`;
document.getElementById('foot').innerHTML =
  `Home-adjusted Elo (K=${DATA.config.k}, home advantage=${DATA.config.hfa} pts). ` +
  `Prequential (one-step-ahead) accuracy on 2000+ decisive matches: ` +
  `<b>${(DATA.metrics.accuracy*100).toFixed(1)}%</b> ` +
  `(log loss ${DATA.metrics.log_loss}, Brier ${DATA.metrics.brier}, n=${DATA.metrics.n}). ` +
  `Source: ESPN. Ratings update when you re-run <code>update_data.py</code> then <code>build_site.py</code>.`;

// ---- ranking table ----
const medals = {1:'🥇',2:'🥈',3:'🥉'};
function dcls(d){return d>0.5?'up':(d<-0.5?'down':'flat')}
function darr(d){return d>0.5?'▲':(d<-0.5?'▼':'–')}
const dotcol = {W:'var(--win)', D:'var(--draw)', L:'var(--loss)'};
let rows='';
DATA.ranking.forEach(r=>{
  const form = r.form.map(f=>{
    const t=`${f.r} v ${f.opp} ${f.gf}-${f.ga} (${f.venue}) · ${f.date}`;
    return `<span class="dot" style="background:${dotcol[f.r]}" title="${t}">${f.r}</span>`;
  }).join('');
  rows += `<tr>
    <td class="rk">${medals[r.rank]?'<span class="medal">'+medals[r.rank]+'</span>':r.rank}</td>
    <td><span class="badge" style="background:${r.color}">${r.abbr}</span><span class="team">${r.team}</span></td>
    <td class="rating">${Math.round(r.rating)}</td>
    <td class="delta ${dcls(r.delta)}">${darr(r.delta)} ${Math.abs(Math.round(r.delta))}</td>
    <td class="form">${form}</td>
    <td class="games">${r.games}</td></tr>`;
});
document.getElementById('tbody').innerHTML = rows;

// ---- recent results ----
document.getElementById('recent').innerHTML = DATA.recent.map(m=>{
  const hw = m.hs>m.as, aw = m.as>m.hs;
  const tag = m.wc ? 'RWC' : m.comp.replace('The ','').replace(' Championship',' C\'ship');
  const nn = m.neutral ? ' <span style="color:var(--muted)">(N)</span>' : '';
  return `<li><span class="d">${m.date}</span>
    <span class="mu"><span class="${hw?'win':''}">${m.home}</span> vs
    <span class="${aw?'win':''}">${m.away}</span>${nn}</span>
    <span class="sc">${m.hs}–${m.as}</span>
    <span class="tag">${tag}</span></li>`;
}).join('');

// ---- win probability calculator ----
(function(){
  const MODEL = DATA.model;
  const rmap = {}; DATA.ranking.forEach(r => rmap[r.team] = r);
  const alpha = DATA.ranking.slice().sort((a,b)=>a.team.localeCompare(b.team));
  const byRank = DATA.ranking.slice().sort((a,b)=>a.rank-b.rank);
  const selA = document.getElementById('teamA'), selB = document.getElementById('teamB');
  selA.innerHTML = alpha.map(t=>`<option>${t.team}</option>`).join('');
  selB.innerHTML = alpha.map(t=>`<option>${t.team}</option>`).join('');
  selA.value = byRank[0].team; selB.value = byRank[1].team;
  let venue = 'N';

  function labels(){
    document.querySelector('#venue [data-v="A"]').textContent = rmap[selA.value].abbr + ' at home';
    document.querySelector('#venue [data-v="B"]').textContent = rmap[selB.value].abbr + ' at home';
  }
  function compute(){
    const A = rmap[selA.value], B = rmap[selB.value];
    const adjA = A.rating + (venue==='A'?MODEL.hfa:0);
    const adjB = B.rating + (venue==='B'?MODEL.hfa:0);
    const diff = adjA - adjB;
    const eA = 1/(1+Math.pow(10,-diff/MODEL.scale));           // expected score A
    let pd = MODEL.draw_a*Math.exp(-MODEL.draw_b*Math.abs(diff));
    pd = Math.min(pd, 2*Math.min(eA,1-eA)*0.95);
    let pA = Math.max(0, eA - pd/2), pB = Math.max(0, (1-eA) - pd/2);
    const margin = MODEL.margin_slope*diff;
    const pct = x => (x*100).toFixed(0)+'%';
    document.getElementById('probbar').innerHTML =
      `<div style="width:${pA*100}%;background:${A.color}">${pA>0.1?A.abbr+' '+pct(pA):''}</div>`+
      `<div style="width:${pd*100}%;background:#94a3b8">${pd>0.07?pct(pd):''}</div>`+
      `<div style="width:${pB*100}%;background:${B.color}">${pB>0.1?B.abbr+' '+pct(pB):''}</div>`;
    document.getElementById('probleg').innerHTML =
      `<span>${A.team} <b>${pct(pA)}</b></span><span>Draw <b>${pct(pd)}</b></span><span>${B.team} <b>${pct(pB)}</b></span>`;
    const fav = margin>0?A:B, mg=Math.abs(margin);
    document.getElementById('pred').textContent =
      mg<1.5 ? 'Predicted: too close to call' : `Predicted: ${fav.team} by ~${Math.round(mg)}`;
  }
  function onChange(){
    if(selA.value===selB.value){
      const alt = alpha.find(t=>t.team!==selA.value); selB.value = alt.team;
    }
    labels(); compute();
  }
  selA.onchange = onChange; selB.onchange = onChange;
  document.querySelectorAll('#venue button').forEach(b=>b.onclick=()=>{
    venue = b.dataset.v;
    document.querySelectorAll('#venue button').forEach(x=>x.classList.toggle('active',x===b));
    compute();
  });
  labels(); compute();
})();

// ---- range buttons ----
const RANGES = [[1871,'All'],[1990,'1990+'],[2000,'2000+'],[2015,'2015+']];
document.getElementById('ranges').innerHTML = RANGES.map(([y,l])=>
  `<button class="rangebtn${y===range?' active':''}" data-y="${y}">${l}</button>`).join('');
document.querySelectorAll('.rangebtn').forEach(b=>b.onclick=()=>{
  range=+b.dataset.y;
  document.querySelectorAll('.rangebtn').forEach(x=>x.classList.toggle('active',x===b));
  draw();
});

// ---- legend ----
document.getElementById('legend').innerHTML = TEAMS.map(t=>
  `<span class="chip" data-t="${t}"><span class="sw" style="background:${META[t].color}"></span>${META[t].abbr}</span>`).join('');
document.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{
  const t=c.dataset.t;
  if(hidden.has(t)){hidden.delete(t);c.classList.remove('off');}
  else{hidden.add(t);c.classList.add('off');}
  draw();
});

// ---- chart ----
const svg = document.getElementById('chart');
const tip = document.getElementById('tip');
const M = {t:14,r:14,b:26,l:42}, H=420;

function visible(){
  const idx=[]; DATA.history.years.forEach((y,i)=>{ if(y>=range) idx.push(i); });
  return idx;
}
function draw(){
  const W = svg.clientWidth || svg.parentNode.clientWidth || 820;
  const idx = visible();
  const years = idx.map(i=>DATA.history.years[i]);
  let lo=Infinity, hi=-Infinity;
  TEAMS.forEach(t=>{ if(hidden.has(t)||!DATA.history.series[t])return;
    idx.forEach(i=>{const v=DATA.history.series[t][i]; if(v!=null){lo=Math.min(lo,v);hi=Math.max(hi,v);}});});
  if(!isFinite(lo)){lo=1000;hi=1900;}
  lo=Math.floor((lo-25)/50)*50; hi=Math.ceil((hi+25)/50)*50;
  const n=years.length;
  const xOf=k=> M.l+(W-M.l-M.r)*(n<=1?0.5:k/(n-1));
  const yOf=v=> M.t+(H-M.t-M.b)*(1-(v-lo)/(hi-lo));

  let s='';
  for(let g=lo; g<=hi; g+=100){
    s+=`<line class="grid" x1="${M.l}" y1="${yOf(g)}" x2="${W-M.r}" y2="${yOf(g)}"/>`;
    s+=`<text class="ylab" x="${M.l-6}" y="${yOf(g)+3}">${g}</text>`;
  }
  const step=Math.max(1,Math.ceil(n/9));
  years.forEach((yr,k)=>{ if(k%step===0) s+=`<text class="xlab" x="${xOf(k)}" y="${H-8}">${yr}</text>`; });
  TEAMS.forEach(t=>{
    if(hidden.has(t)||!DATA.history.series[t])return;
    let d='',pen=false;
    idx.forEach((i,k)=>{const v=DATA.history.series[t][i];
      if(v==null){pen=false;return;} d+=(pen?'L':'M')+xOf(k).toFixed(1)+' '+yOf(v).toFixed(1)+' '; pen=true;});
    if(d) s+=`<path d="${d}" fill="none" stroke="${META[t].color}" stroke-width="2.1" stroke-linejoin="round"/>`;
  });
  s+=`<line id="vl" class="vline" x1="0" y1="${M.t}" x2="0" y2="${H-M.b}" style="opacity:0"/>`;
  svg.setAttribute('width',W); svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  svg.innerHTML=s;

  svg.onmousemove=ev=>{
    const rect=svg.getBoundingClientRect();
    const px=(ev.clientX-rect.left)*(W/rect.width);
    if(n<1)return;
    let k=Math.round((px-M.l)/((W-M.l-M.r)/(Math.max(1,n-1))));
    k=Math.max(0,Math.min(n-1,k));
    const vl=document.getElementById('vl');
    vl.setAttribute('x1',xOf(k)); vl.setAttribute('x2',xOf(k)); vl.style.opacity=1;
    const items=TEAMS.filter(t=>!hidden.has(t)&&DATA.history.series[t]&&DATA.history.series[t][idx[k]]!=null)
      .map(t=>({t,v:DATA.history.series[t][idx[k]]})).sort((a,b)=>b.v-a.v);
    tip.innerHTML=`<div class="ty">${years[k]}</div>`+items.map(o=>
      `<div class="row"><span><span class="sw" style="background:${META[o.t].color}"></span>${META[o.t].abbr}</span><b>${Math.round(o.v)}</b></div>`).join('');
    let tx=xOf(k)+14; if(tx> W-170) tx=xOf(k)-170;
    tip.style.transform=`translate(${tx}px,${M.t+6}px)`;
  };
  svg.onmouseleave=()=>{tip.style.transform='translate(-9999px,-9999px)';
    const vl=document.getElementById('vl'); if(vl)vl.style.opacity=0;};
}
draw();
let rt; window.addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(draw,120);});
</script>
<!--WIDGET-END-->
</body>
</html>
"""


if __name__ == "__main__":
    main()

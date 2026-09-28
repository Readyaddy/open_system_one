"""Live training dashboard -- a real local HTTP endpoint, not a static export.

  python scripts/dashboard_server.py [--port 8770] [--run-dir <checkpoint dir>]

Then open http://localhost:8770 . The page polls /api/state every few seconds
and re-renders; nothing is cached, so it always reflects the metrics JSON and
log file as they are on disk right now.

Serves:
  /                 the dashboard page
  /api/state        JSON: parsed metrics + parsed log tail + GPU + process state
  /api/log?n=400    raw log tail
"""
import argparse
import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STEP_RE = re.compile(
    r"\[ep (\d+)/(\d+)\]\s+(\d+)/(\d+)\s+\((\d+)/(\d+)\)\s+loss\s+([\d.]+)\s+"
    r"mono\s+([\d.]+)\s+lr\s+([\d.e+-]+)\s+([\d.]+) it/s\s+([\d.]+)GiB\s+ETA\s+(\d+)m"
    r"(?:\s+\[data_wait (\d+)% \| alloc_retries (\d+)\])?")

ARGS = None


def gpu_stats():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4)
        if out.returncode != 0:
            return None
        f = [x.strip() for x in out.stdout.strip().split("\n")[0].split(",")]
        return {"mem_used": float(f[0]), "mem_total": float(f[1]),
                "util": float(f[2]), "temp": float(f[3]),
                "power": float(f[4]) if f[4] not in ("[N/A]", "") else None}
    except Exception:
        return None


def find_files(run_dir):
    metrics, log = None, None
    if not os.path.isdir(run_dir):
        return metrics, log
    cands = []
    for fn in os.listdir(run_dir):
        p = os.path.join(run_dir, fn)
        if not os.path.isfile(p):
            continue
        if fn.endswith("_metrics.json") or fn == "metrics.json":
            cands.append((os.path.getmtime(p), "m", p))
        elif fn.endswith(".log"):
            # Skip this server's own log -- it lives in the same directory and is
            # touched on every request, so "most recently modified .log" would
            # always select it and the dashboard would display itself.
            if fn == "dashboard.log":
                continue
            cands.append((os.path.getmtime(p), "l", p))
    for _, kind, p in sorted(cands, reverse=True):
        if kind == "m" and metrics is None:
            metrics = p
        if kind == "l" and log is None:
            log = p
    # stdout goes to train.log; stderr (library warnings) may land in a newer
    # sibling like train.err.log, which must not win the "most recent" race.
    if os.path.exists(os.path.join(run_dir, "train.log")):
        log = os.path.join(run_dir, "train.log")
    return metrics, log


def read_log_tail(path, n=400):
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()[-n:]
    except Exception:
        return []


def parse_state():
    run_dir = ARGS.run_dir
    metrics_path, log_path = find_files(run_dir)

    metrics = {}
    if metrics_path and os.path.exists(metrics_path):
        try:
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}

    lines = read_log_tail(log_path, 600)
    steps = []
    for ln in lines:
        m = STEP_RE.search(ln)
        if m:
            steps.append({
                "epoch": int(m.group(1)), "epochs_total": int(m.group(2)),
                "step": int(m.group(3)), "steps_total": int(m.group(4)),
                "global_step": int(m.group(5)), "global_total": int(m.group(6)),
                "loss": float(m.group(7)), "mono": float(m.group(8)),
                "lr": float(m.group(9)), "it_s": float(m.group(10)),
                "mem": float(m.group(11)), "eta_min": int(m.group(12)),
                "data_wait": int(m.group(13)) if m.group(13) else None,
                "alloc_retries": int(m.group(14)) if m.group(14) else None,
            })

    alive = False
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"],
                              capture_output=True, text=True, timeout=5)
        alive = "python.exe" in out.stdout
    except Exception:
        pass

    mtime = os.path.getmtime(log_path) if log_path and os.path.exists(log_path) else 0
    return {
        "run_dir": run_dir, "metrics_file": metrics_path, "log_file": log_path,
        "metrics": metrics, "steps": steps[-300:], "latest": steps[-1] if steps else None,
        "log_tail": [l.rstrip("\n") for l in lines[-200:]],
        "gpu": gpu_stats(), "process_alive": alive,
        "log_age_sec": round(time.time() - mtime, 1) if mtime else None,
        "server_time": time.strftime("%H:%M:%S"),
    }


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>exp7 live</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{--bg:#0b0e14;--surface:#131722;--surface2:#171c29;--border:#232838;--text:#e6e9f0;
--dim:#8b93a7;--faint:#4d5568;--accent:#5fd0ff;--warm:#ff9d5c;--good:#4ade80;--warn:#fbbf24;--bad:#f87171;
color-scheme:dark}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg)}
body{font-family:"IBM Plex Sans",system-ui,sans-serif;color:var(--text);padding:20px;font-size:14px}
.mono{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
.wrap{max-width:1250px;margin:0 auto;display:flex;flex-direction:column;gap:16px}
header{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap;
border-bottom:1px solid var(--border);padding-bottom:14px}
h1{font-size:19px;margin:0;font-weight:700}
h1 span{color:var(--dim);font-weight:500;font-size:14px}
.pill{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;padding:3px 9px;
border-radius:999px;border:1px solid var(--border);background:var(--surface2);color:var(--dim)}
.pill.live{color:var(--good);border-color:#2e6b45}.pill.stale{color:var(--warn);border-color:#6b5a24}
.pill.dead{color:var(--bad);border-color:#6b2e2e}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.k{color:var(--faint);font-size:10px;text-transform:uppercase;letter-spacing:.05em}
.v{font-size:20px;font-weight:600;margin-top:3px}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px 18px}
.panel h2{margin:0 0 10px;font-size:13px;font-weight:600}
.bar{height:7px;border-radius:4px;background:var(--surface2);overflow:hidden;margin-top:8px}
.bar>div{height:100%;background:var(--accent)}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{text-align:right;padding:6px 9px;border-bottom:1px solid var(--border);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--faint);font-size:10px;text-transform:uppercase;letter-spacing:.04em}
td{font-family:"IBM Plex Mono",monospace}
.pos{color:var(--good)}.neg{color:var(--bad)}.zero{color:var(--faint)}
#log{background:#05070c;border:1px solid var(--border);border-radius:8px;padding:11px 13px;
height:330px;overflow-y:auto;font-size:11.5px;line-height:1.55;white-space:pre-wrap;word-break:break-word}
.l-ep{color:var(--good)}.l-best{color:var(--warn);font-weight:600}.l-warn{color:var(--bad)}.l-d{color:var(--dim)}
svg{width:100%;height:auto;display:block;overflow:visible}
.ax{fill:var(--faint);font-size:9px;font-family:"IBM Plex Mono",monospace}
.gl{stroke:var(--border);stroke-width:1}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:820px){.two{grid-template-columns:1fr}}
footer{color:var(--faint);font-size:11px;text-align:center}
</style></head><body><div class="wrap">
<header>
  <h1>exp7 live <span id="rd"></span></h1>
  <div style="display:flex;gap:8px;align-items:center">
    <span class="pill" id="status">connecting</span>
    <span class="mono" style="color:var(--faint);font-size:11px" id="clock"></span>
  </div>
</header>
<div class="grid" id="stats"></div>
<div class="panel"><h2>Progress</h2><div id="prog"></div></div>
<div class="two">
  <div class="panel"><h2>Training loss</h2><div id="chart-loss"></div></div>
  <div class="panel"><h2>GPU</h2><div id="chart-gpu"></div></div>
</div>
<div class="two">
  <div class="panel"><h2>Validation score by epoch (model selection)</h2><div id="chart-val"></div></div>
  <div class="panel"><h2>In-distribution validation (latest)</h2><div id="valtab"></div></div>
</div>
<div class="panel"><h2>Recurrent depth &times; required reasoning hops</h2>
  <div id="qdep"></div></div>
<div class="panel"><h2>Hard-111 per-depth accuracy (95% CI)</h2><div id="h111"></div></div>
<div class="panel"><h2>Log</h2><div id="log" class="mono"></div></div>
<footer id="foot"></footer>
</div>
<script>
let pinned=true;
const logEl=document.getElementById('log');
logEl.addEventListener('scroll',()=>{pinned=logEl.scrollTop+logEl.clientHeight>=logEl.scrollHeight-30;});
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function line(t,v,u=''){return `<div class="card"><div class="k">${t}</div><div class="v mono">${v}${u}</div></div>`}

function chart(el,series,fmt,h=150){
  const W=520,padL=44,padR=10,padT=10,padB=20;
  const xs=series.flatMap(s=>s.pts.map(p=>p[0])), ys=series.flatMap(s=>s.pts.map(p=>p[1]));
  if(!xs.length){el.innerHTML='<div style="color:var(--faint);font-size:12px;padding:24px;text-align:center">no data yet</div>';return}
  let x0=Math.min(...xs),x1=Math.max(...xs,x0+1),y0=Math.min(...ys),y1=Math.max(...ys);
  if(y1-y0<1e-9){y1+=1;y0-=1}
  const pd=(y1-y0)*.12;y0-=pd;y1+=pd;
  const sx=x=>padL+(x-x0)/(x1-x0)*(W-padL-padR), sy=y=>h-padB-(y-y0)/(y1-y0)*(h-padT-padB);
  let g='';
  for(let i=0;i<=4;i++){const gy=padT+i/4*(h-padT-padB),val=y1-i/4*(y1-y0);
    g+=`<line class="gl" x1="${padL}" x2="${W-padR}" y1="${gy}" y2="${gy}"/>`;
    g+=`<text class="ax" x="${padL-6}" y="${gy+3}" text-anchor="end">${fmt(val)}</text>`}
  for(const s of series){
    const d=s.pts.map((p,i)=>`${i?'L':'M'} ${sx(p[0]).toFixed(1)} ${sy(p[1]).toFixed(1)}`).join(' ');
    g+=`<path d="${d}" fill="none" stroke="${s.c}" stroke-width="1.8" stroke-linejoin="round"/>`;
  }
  g+=`<text class="ax" x="${padL}" y="${h-5}">${x0}</text><text class="ax" x="${W-padR}" y="${h-5}" text-anchor="end">${x1}</text>`;
  el.innerHTML=`<svg viewBox="0 0 ${W} ${h}" preserveAspectRatio="xMidYMid meet">${g}</svg>`;
}

async function tick(){
  let s;
  try{s=await (await fetch('/api/state',{cache:'no-store'})).json()}
  catch(e){document.getElementById('status').className='pill dead';
           document.getElementById('status').textContent='server unreachable';return}
  const st=document.getElementById('status');
  const fresh=s.log_age_sec!==null&&s.log_age_sec<120;
  st.className='pill '+(fresh?'live':(s.process_alive?'stale':'dead'));
  st.textContent=fresh?'training':(s.process_alive?'idle / evaluating':'not running');
  document.getElementById('clock').textContent=s.server_time;
  document.getElementById('rd').textContent=(s.metrics.experiment||'')+' — '+(s.run_dir||'');

  const L=s.latest, g=s.gpu, m=s.metrics;
  let h='';
  if(L){
    h+=line('Epoch',`${L.epoch}/${L.epochs_total}`);
    h+=line('Step',`${L.step}/${L.steps_total}`);
    h+=line('Loss',L.loss.toFixed(4));
    h+=line('Monotonic',L.mono.toFixed(4));
    h+=line('LR',L.lr.toExponential(2));
    h+=line('Speed',L.it_s.toFixed(2),' it/s');
    h+=line('ETA',L.eta_min,' min');
    if(L.data_wait!==null&&L.data_wait!==undefined)h+=line('Data wait',L.data_wait,'%');
  }
  if(g){h+=line('GPU mem',(g.mem_used/1024).toFixed(1),`/${(g.mem_total/1024).toFixed(0)} GiB`);
        h+=line('GPU util',g.util.toFixed(0),'%');
        h+=line('Temp',g.temp.toFixed(0),'°C');}
  document.getElementById('stats').innerHTML=h;

  if(L){const p=L.global_step/Math.max(1,L.global_total)*100;
    document.getElementById('prog').innerHTML=
      `<div style="display:flex;justify-content:space-between;color:var(--dim);font-size:12px">
       <span class="mono">${L.global_step} / ${L.global_total} steps</span><span class="mono">${p.toFixed(1)}%</span></div>
       <div class="bar"><div style="width:${p}%"></div></div>`;}

  chart(document.getElementById('chart-loss'),
        [{c:'#5fd0ff',pts:s.steps.map(d=>[d.global_step,d.loss])}],v=>v.toFixed(2));
  chart(document.getElementById('chart-gpu'),
        [{c:'#ff9d5c',pts:s.steps.map(d=>[d.global_step,d.mem])}],v=>v.toFixed(1)+'G');

  // validation score history + latest per-group table
  const vpts=[];
  if(m.initial&&m.initial.val_score!==undefined)vpts.push([0,m.initial.val_score*100]);
  for(const e of (m.evals||[]))if(e.val_score!==undefined)vpts.push([e.epoch,e.val_score*100]);
  chart(document.getElementById('chart-val'),[{c:'#4ade80',pts:vpts}],v=>v.toFixed(1)+'%');
  let vsrc=null,vtag='';
  const evs0=(m.evals||[]);
  if(evs0.length&&evs0[evs0.length-1].val){vsrc=evs0[evs0.length-1].val;vtag='epoch '+evs0[evs0.length-1].epoch}
  else if(m.initial&&m.initial.val){vsrc=m.initial.val;vtag='baseline'}
  const vt=document.getElementById('valtab');
  if(vsrc){
    let t='<table><thead><tr><th>group</th><th>n</th><th>k=1</th><th>k=6</th><th>pred T/F/U</th></tr></thead><tbody>';
    for(const [g,v] of Object.entries(vsrc)){ if(g.startsWith('_'))continue;
      const a=v.accs, ph=v.pred_hist;
      let hs='';
      if(ph){const top=Math.max(ph.True,ph.False,ph.Unknown);
        hs=`<span class="${top>0.85?'neg':''}">${(ph.True*100).toFixed(0)}/${(ph.False*100).toFixed(0)}/${(ph.Unknown*100).toFixed(0)}</span>`}
      t+=`<tr><td>${g}</td><td>${v.n}</td><td>${(a[0]*100).toFixed(1)}</td><td>${(a[a.length-1]*100).toFixed(1)}</td><td>${hs}</td></tr>`}
    const bl=vsrc._pw_by_length||{};
    t+=`</tbody></table><div style="color:var(--faint);font-size:11px;margin-top:8px">ProofWriter by context length: `+
       Object.entries(bl).filter(([k,v])=>v!==null).map(([k,v])=>`${k} ${(v*100).toFixed(1)}%`).join(' · ')+
       ` (${vtag}). Red histogram = one class &gt;85% of predictions (collapse).</div>`;
    vt.innerHTML=t;
  } else vt.innerHTML='<div style="color:var(--faint);font-size:12px;padding:18px;text-align:center">waiting for baseline eval</div>';

  // qdep grid from the most recent eval that has one
  const evs=(m.evals||[]);
  let qsrc=null,qtag='';
  for(let i=evs.length-1;i>=0;i--){if(evs[i].qdep_grid&&Object.keys(evs[i].qdep_grid).length){qsrc=evs[i].qdep_grid;qtag='epoch '+evs[i].epoch;break}}
  if(!qsrc&&m.initial&&m.initial.qdep_grid&&Object.keys(m.initial.qdep_grid).length){qsrc=m.initial.qdep_grid;qtag='baseline'}
  const qe=document.getElementById('qdep');
  if(qsrc){
    const order=['0','1','2-3','4+'].filter(b=>qsrc[b]);
    const K=qsrc[order[0]].accs.length;
    let t='<table><thead><tr><th>hops</th><th>n</th>';
    for(let i=0;i<K;i++)t+=`<th>k=${i+1}</th>`;
    t+='<th>k6−k1</th></tr></thead><tbody>';
    for(const b of order){const a=qsrc[b].accs,d=(a[K-1]-a[0])*100;
      t+=`<tr><td>${b}</td><td>${qsrc[b].n}</td>`;
      for(const x of a)t+=`<td>${(x*100).toFixed(1)}</td>`;
      t+=`<td class="${d>0.05?'pos':(d<-0.05?'neg':'zero')}">${d>=0?'+':''}${d.toFixed(1)}</td></tr>`}
    t+='</tbody></table><div style="color:var(--faint);font-size:11px;margin-top:8px">'+
       'A working depth mechanism shows a larger k6−k1 on higher hop counts. All zeros = flat loop. ('+qtag+')</div>';
    qe.innerHTML=t;
  } else qe.innerHTML='<div style="color:var(--faint);font-size:12px;padding:18px;text-align:center">waiting for first eval</div>';

  // hard-111
  let hsrc=null,htag='';
  if(evs.length){hsrc=evs[evs.length-1];htag='epoch '+hsrc.epoch}
  else if(m.initial){hsrc=m.initial;htag='baseline'}
  const he=document.getElementById('h111');
  if(hsrc&&hsrc.accs){
    let t='<table><thead><tr><th>depth</th><th>acc</th><th>95% CI</th></tr></thead><tbody>';
    hsrc.accs.forEach((a,i)=>{const ci=(hsrc.cis&&hsrc.cis[i])||[0,0];
      t+=`<tr><td>k=${i+1}</td><td>${(a*100).toFixed(2)}%</td><td style="color:var(--faint)">${(ci[0]*100).toFixed(1)} – ${(ci[1]*100).toFixed(1)}</td></tr>`});
    const sp=(Math.max(...hsrc.accs)-Math.min(...hsrc.accs))*100;
    t+=`</tbody></table><div style="color:var(--faint);font-size:11px;margin-top:8px">spread ${sp.toFixed(2)} pts (${htag}) — under ~1.8 pts is within one example on n=111</div>`;
    he.innerHTML=t;
  } else he.innerHTML='<div style="color:var(--faint);font-size:12px;padding:18px;text-align:center">waiting</div>';

  const cls=l=>/new best|\*\*\*/.test(l)?'l-best':(/^epoch |^--- |VAL SCORE|^\[epoch/.test(l)?'l-ep':(/WARNING|Error|Traceback|!!/.test(l)?'l-warn':'l-d'));
  logEl.innerHTML=s.log_tail.map(l=>`<div class="${cls(l)}">${esc(l)}</div>`).join('');
  if(pinned)logEl.scrollTop=logEl.scrollHeight;
  document.getElementById('foot').textContent=
    (s.log_file||'no log')+'  ·  updated '+(s.log_age_sec===null?'?':s.log_age_sec+'s ago');
}
tick();setInterval(tick,4000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        try:
            if self.path.startswith("/api/state"):
                self._send(200, json.dumps(parse_state()), "application/json")
            elif self.path.startswith("/api/log"):
                _, log = find_files(ARGS.run_dir)
                self._send(200, "".join(read_log_tail(log, 400)), "text/plain; charset=utf-8")
            elif self.path in ("/", "/index.html"):
                self._send(200, PAGE, "text/html; charset=utf-8")
            else:
                self._send(404, "not found", "text/plain")
        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self._send(500, json.dumps({"error": str(e)}), "application/json")
            except Exception:
                pass


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--run-dir", type=str, default=None,
                     help="Checkpoint dir to watch. Default: the most recently updated run under "
                          "D:\\projects\\JEPA\\checkpoints that has a *_metrics.json.")
    ARGS = ap.parse_args()
    if ARGS.run_dir is None:
        root = r"D:\projects\JEPA\checkpoints"
        cands = []
        for d in os.listdir(root):
            full = os.path.join(root, d)
            if os.path.isdir(full):
                ms = [os.path.join(full, f) for f in os.listdir(full) if f.endswith("_metrics.json")]
                if ms:
                    cands.append((max(os.path.getmtime(x) for x in ms), full))
        ARGS.run_dir = max(cands)[1] if cands else root

    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"dashboard: http://{ARGS.host}:{ARGS.port}   watching {ARGS.run_dir}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

"""Local frontend for the deployed FAQ chatbot.

    python scripts/serve.py
    open http://localhost:8000

Serves the chat UI and proxies /api/chat to the deployed Lambda using the AWS
SDK (`lambda:InvokeFunction`) rather than the public Function URL. The URL is
blocked on this account by an organisation service control policy; invoking
through the SDK reaches the same function, runs the same model and writes the
same DynamoDB records — only the transport differs.

Because the page and the API share an origin, there is no CORS involved.
"""
from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

REGION = os.environ.get("AWS_REGION", "us-east-1")
FUNCTION = os.environ.get("FAQ_FUNCTION", "faq-chatbot")
PORT = int(os.environ.get("PORT", "8000"))

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AWS FAQ Chatbot</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0F1729; --panel:#1A2438; --panel2:#212D45; --line:#2B3854;
    --text:#E7ECF5; --muted:#8B98B4; --dim:#64708C;
    --accent:#5B8DEF; --accent2:#4478E0; --good:#2FBF8F; --warn:#E5834A; --bad:#E2574C;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0; background:var(--bg); color:var(--text);
       font-family:Inter,system-ui,sans-serif; font-size:14px}

  /* top bar */
  .top{background:var(--panel); padding:0 22px; display:flex; align-items:center;
       gap:26px; height:60px; border-bottom:1px solid var(--line)}
  .brand{font-weight:700; font-size:18px; letter-spacing:-.01em}
  .brand em{font-style:normal; color:var(--accent)}
  .top .meta{margin-left:auto; display:flex; align-items:center; gap:18px;
             font-size:12px; color:var(--muted)}
  .pill{display:inline-flex; align-items:center; gap:7px; background:var(--panel2);
        padding:6px 12px; border-radius:99px}
  .dot{width:7px; height:7px; border-radius:50%; background:var(--good)}
  .dot.off{background:var(--bad)}

  /* layout */
  .grid{display:grid; grid-template-columns:260px minmax(0,1fr) 300px;
        gap:18px; padding:18px; align-items:start}
  .card{background:var(--panel); border:1px solid var(--line); border-radius:12px}
  .card h2{font-size:13px; font-weight:600; margin:0; padding:16px 18px;
           border-bottom:1px solid var(--line); display:flex; justify-content:space-between}
  .card h2 span{color:var(--dim); font-weight:400}

  /* topics rail */
  .topics{padding:10px}
  .topic{width:100%; text-align:left; background:transparent; border:none; color:var(--text);
         font:inherit; padding:10px 12px; border-radius:8px; cursor:pointer; display:block}
  .topic:hover,.topic:focus-visible{background:var(--panel2); outline:none}
  .topic b{display:block; font-weight:500; font-size:13px}
  .topic small{color:var(--dim); font-size:11.5px}
  .rail-note{padding:12px 18px 16px; color:var(--dim); font-size:11.5px; line-height:1.5;
             border-top:1px solid var(--line)}

  /* chat */
  .chat{display:flex; flex-direction:column; height:calc(100vh - 96px); min-height:520px}
  .chat-head{display:flex; align-items:center; gap:12px; padding:14px 18px;
             border-bottom:1px solid var(--line)}
  .av{width:38px; height:38px; border-radius:10px; background:var(--accent);
      display:grid; place-items:center; font-weight:600; font-size:11.5px;
      letter-spacing:.02em; color:#fff; flex:none}
  .av.you{background:var(--panel2); color:var(--muted)}
  .chat-head b{display:block; font-size:14px}
  .chat-head small{color:var(--dim); font-size:11.5px}
  #log{flex:1; overflow-y:auto; padding:20px 18px; display:flex; flex-direction:column; gap:16px}
  #log::-webkit-scrollbar{width:8px} #log::-webkit-scrollbar-thumb{background:var(--line); border-radius:4px}

  .msg{display:flex; gap:11px; max-width:76%}
  .msg.me{margin-left:auto; flex-direction:row-reverse}
  .bub{background:var(--panel2); padding:11px 14px; border-radius:12px; line-height:1.55;
       border-top-left-radius:4px}
  .msg.me .bub{background:var(--accent); color:#fff; border-top-left-radius:12px;
               border-top-right-radius:4px}
  .msg.refused .bub{background:rgba(226,87,76,.12); border:1px solid rgba(226,87,76,.35)}
  .stamp{font-size:11px; color:var(--dim); margin-bottom:5px}
  .msg.me .stamp{text-align:right}
  .cites{margin-top:7px; font-size:11.5px; color:var(--dim)}

  .dots{display:inline-flex; gap:5px; padding:4px 2px}
  .dots i{width:6px; height:6px; border-radius:50%; background:var(--muted); animation:blink 1.3s infinite}
  .dots i:nth-child(2){animation-delay:.18s} .dots i:nth-child(3){animation-delay:.36s}
  @keyframes blink{0%,60%,100%{opacity:.3;transform:translateY(0)} 30%{opacity:1;transform:translateY(-4px)}}
  .caret{display:inline-block; width:2px; height:1em; background:var(--accent);
         vertical-align:-2px; animation:caret .9s steps(1) infinite}
  @keyframes caret{0%,50%{opacity:1}51%,100%{opacity:0}}

  .rate{margin-top:9px; display:flex; align-items:center; gap:7px; font-size:11.5px; color:var(--dim)}
  .rate button{background:var(--panel); border:1px solid var(--line); color:var(--muted);
               padding:3px 10px; border-radius:99px; font-size:11.5px; cursor:pointer}
  .rate button:hover{border-color:var(--accent); color:var(--text)}
  .rate button[aria-pressed=true]{background:var(--accent); border-color:var(--accent); color:#fff}
  .rate .ok{color:var(--good)}

  .clarify{margin-top:10px; padding:11px 13px; background:var(--panel);
           border:1px solid var(--line); border-radius:10px}
  .clarify p{margin:0 0 8px; font-size:12.5px; color:var(--muted)}
  .clarify button{background:var(--panel2); border:1px solid var(--line); color:var(--text);
                  padding:6px 12px; border-radius:8px; font-size:12.5px; margin:0 6px 0 0; cursor:pointer}
  .clarify button:hover{border-color:var(--accent)}

  /* composer */
  .composer{border-top:1px solid var(--line); padding:14px 18px; display:flex; gap:10px}
  .composer input{flex:1; background:var(--panel2); border:1px solid transparent; color:var(--text);
                  font:inherit; padding:12px 15px; border-radius:10px}
  .composer input::placeholder{color:var(--dim)}
  .composer input:focus{outline:none; border-color:var(--accent)}
  .composer button{width:44px; height:44px; border:none; border-radius:50%; background:var(--accent);
                   color:#fff; font-size:16px; cursor:pointer; flex:none}
  .composer button:disabled{opacity:.4; cursor:default}
  .composer button:focus-visible{outline:2px solid #fff; outline-offset:2px}

  /* diagnostics */
  .diag{padding:16px 18px}
  .empty{color:var(--dim); font-size:12.5px; line-height:1.6}
  .metric{margin-bottom:15px}
  .metric .k{display:flex; justify-content:space-between; font-size:11.5px;
             color:var(--muted); margin-bottom:6px}
  .metric .k b{color:var(--text); font-weight:600; font-variant-numeric:tabular-nums}
  .bar{height:5px; background:var(--panel2); border-radius:3px; overflow:hidden}
  .bar i{display:block; height:100%; background:var(--accent); border-radius:3px; transition:width .4s}
  .bar i.lo{background:var(--warn)} .bar i.no{background:var(--bad)}
  .badge{display:inline-block; padding:4px 10px; border-radius:6px; font-size:11px; font-weight:500}
  .badge.stop{background:rgba(226,87,76,.15); color:#FF8A80; border:1px solid rgba(226,87,76,.35)}
  .kv{display:grid; grid-template-columns:auto 1fr; gap:8px 12px; margin:16px 0 0;
      padding-top:14px; border-top:1px solid var(--line); font-size:12px}
  .kv dt{color:var(--dim)} .kv dd{margin:0; text-align:right; font-variant-numeric:tabular-nums}
  .alts{margin-top:15px; padding-top:13px; border-top:1px solid var(--line)}
  .alts .t{font-size:11.5px; color:var(--dim); margin-bottom:9px}
  .alts .r{display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px; color:var(--muted)}
  .alts .r b{color:var(--text); font-weight:500; font-variant-numeric:tabular-nums}

  @media (max-width:1100px){ .grid{grid-template-columns:1fr} .chat{height:auto; min-height:460px}
                             #log{max-height:52vh} }
  @media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
</style>
</head>
<body>

<div class="top">
  <div class="brand">AWS FAQ <em>Chatbot</em></div>
  <div class="meta">
    <span class="pill"><i class="dot" id="live"></i><span id="conn">connecting</span></span>
    <span id="rt"></span>
  </div>
</div>

<div class="grid">
  <!-- topics -->
  <section class="card">
    <h2>Topics <span id="tcount"></span></h2>
    <div class="topics" id="topics"></div>
    <p class="rail-note">27 intents across 11 categories, trained on 8,175
       support utterances. Pick one to send it as a question.</p>
  </section>

  <!-- conversation -->
  <section class="card chat">
    <div class="chat-head">
      <div class="av">FAQ</div>
      <div>
        <b>AWS FAQ Chatbot</b>
        <small id="sub">Intent classification and FAQ retrieval on AWS Lambda</small>
      </div>
    </div>
    <div id="log"></div>
    <form class="composer" id="ask">
      <input id="q" type="text" placeholder="Ask about orders, refunds, payments, your account…"
             autocomplete="off" maxlength="300">
      <button id="send" type="submit" aria-label="Send">&#10148;</button>
    </form>
  </section>

  <!-- diagnostics -->
  <aside class="card">
    <h2>What the model did <span id="turns"></span></h2>
    <div class="diag" id="panel">
      <p class="empty">Ask a question and this panel shows the predicted intent,
        classifier confidence, which FAQ was retrieved and how long Lambda took.</p>
    </div>
  </aside>
</div>

<script>
const TOPICS = [
  ['Track an order',      'Where is my order 4562781?'],
  ['Cancel an order',     'I need to cancel my order'],
  ['Change an order',     'I want to change my order'],
  ['Refunds',             'How do I get a refund?'],
  ['Payment problems',    'My payment was declined'],
  ['Payment methods',     'What payment methods do you accept?'],
  ['Reset password',      'How do I reset my password?'],
  ['Create an account',   'I want to create an account'],
  ['Delete an account',   'How do I delete my account?'],
  ['Invoices',            'Where can I find my invoice?'],
  ['Delivery options',    'What shipping options are there?'],
  ['Newsletter',          'Unsubscribe me from the newsletter'],
  ['Contact a human',     'I want to talk to an agent'],
  ['Off topic (test)',    'Who won the world cup in 1998?']
];

const log=document.getElementById('log'), panel=document.getElementById('panel'),
      form=document.getElementById('ask'), input=document.getElementById('q'),
      send=document.getElementById('send'), rail=document.getElementById('topics');
let sessionId=null, turns=0;

const pct=n=>Math.round(n*100);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const clock=()=>new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});

TOPICS.forEach(([label,q])=>{
  const b=document.createElement('button');
  b.type='button'; b.className='topic';
  b.innerHTML='<b>'+esc(label)+'</b><small>'+esc(q)+'</small>';
  b.addEventListener('click',()=>ask(q));
  rail.appendChild(b);
});
document.getElementById('tcount').textContent=TOPICS.length;

function msg(who, inner, refused){
  const d=document.createElement('div');
  d.className='msg '+who+(refused?' refused':'');
  d.innerHTML='<div class="av'+(who==='me'?' you':'')+'">'+(who==='me'?'You':'FAQ')+'</div>'+
              '<div><div class="stamp">'+clock()+'</div><div class="bub">'+inner+'</div></div>';
  log.appendChild(d); log.scrollTop=log.scrollHeight;
  return d;
}

/* Progressive reveal. The model returns the full string at once, so this is
   presentation only: fast (capped near a second) and skipped when the reader
   prefers reduced motion, so it reads as responsive rather than as latency. */
function reveal(el,text){
  if(window.matchMedia('(prefers-reduced-motion: reduce)').matches || text.length>400){
    el.textContent=text; return Promise.resolve();
  }
  return new Promise(done=>{
    const step=Math.max(1,Math.ceil(text.length/85)); let i=0;
    const t=setInterval(()=>{
      i=Math.min(text.length,i+step);
      el.innerHTML=esc(text.slice(0,i))+(i<text.length?'<span class="caret"></span>':'');
      log.scrollTop=log.scrollHeight;
      if(i>=text.length){clearInterval(t); done();}
    },12);
  });
}

/* Thumbs rating -> DynamoDB. This is the 'user satisfaction' figure the
   analytics module averages; without a rating written back there is no data. */
function rate(box,r){
  const bar=document.createElement('div'); bar.className='rate';
  bar.innerHTML='<span>Helpful?</span>'+
    '<button type="button" aria-pressed="false" data-v="1">Yes</button>'+
    '<button type="button" aria-pressed="false" data-v="0">No</button>';
  box.appendChild(bar);
  bar.querySelectorAll('button').forEach(b=>b.addEventListener('click',async()=>{
    bar.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed','false'));
    b.setAttribute('aria-pressed','true');
    try{
      await fetch('/api/feedback',{method:'POST',headers:{'content-type':'application/json'},
        body:JSON.stringify({request_id:r.request_id,sessionId:sessionId,
                             helpful:Number(b.dataset.v),query:r.query,intent:r.intent})});
      const s=bar.querySelector('span'); s.textContent='Recorded'; s.className='ok';
    }catch(e){ bar.querySelector('span').textContent='Could not save'; }
  }));
}

/* When the top two intents sit within 15 points the classifier is unsure
   rather than wrong -- the order-lifecycle intents genuinely overlap. Ask
   instead of guessing. */
function clarify(box,r){
  const a=r.alternatives||[];
  if(r.is_fallback||a.length<2||a[0].confidence-a[1].confidence>0.15) return;
  const c=document.createElement('div'); c.className='clarify';
  c.innerHTML='<p>Those two are close. Which did you mean?</p>';
  a.slice(0,2).forEach(x=>{
    const b=document.createElement('button'); b.type='button';
    b.textContent=x.intent.replace(/_/g,' ');
    b.addEventListener('click',()=>{c.remove(); ask(x.intent.replace(/_/g,' '));});
    c.appendChild(b);
  });
  box.appendChild(c);
}

function draw(r){
  const conf=r.intent_confidence||0, sim=r.retrieval_score||0;
  const cls=v=>v>=.6?'':v>=.35?'lo':'no';
  let h='';
  if(r.is_fallback){
    h+='<p style="margin:0 0 14px"><span class="badge stop">No answer given</span></p>'+
       '<p style="margin:0 0 16px;color:var(--muted);font-size:12px">Gate: '+
       esc(r.fallback_reason||'unknown')+'</p>';
  }
  h+=bar('Intent confidence',conf,pct(conf)+'%',cls(conf))+
     bar('Retrieval similarity',sim,sim.toFixed(3),cls(sim))+
     bar('Vocabulary match',r.lexical_coverage||0,pct(r.lexical_coverage||0)+'%',cls(r.lexical_coverage||0));
  h+='<dl class="kv">'+kv('Intent',r.intent||'—')+kv('FAQ',r.faq_id||'—')+
     kv('Source',r.answer_mode)+kv('Lambda',(r.latency_ms||0).toFixed(1)+' ms');
  const slots=(r.entities&&r.entities.slots)||{};
  for(const k in slots) h+=kv(k.replace(/_/g,' '),slots[k]);
  h+='</dl>';
  if((r.alternatives||[]).length>1){
    h+='<div class="alts"><div class="t">Runner-up intents</div>';
    r.alternatives.slice(1).forEach(a=>h+='<div class="r"><span>'+esc(a.intent)+
      '</span><b>'+pct(a.confidence)+'%</b></div>');
    h+='</div>';
  }
  panel.innerHTML=h;
  document.getElementById('turns').textContent='turn '+turns;
}
const bar=(label,v,shown,cls)=>'<div class="metric"><div class="k"><span>'+label+
  '</span><b>'+shown+'</b></div><div class="bar"><i class="'+cls+'" style="width:'+
  Math.max(2,pct(v))+'%"></i></div></div>';
const kv=(k,v)=>'<dt>'+esc(k)+'</dt><dd>'+esc(v)+'</dd>';

async function ask(text){
  if(!text.trim()||send.disabled) return;
  msg('me',esc(text)); input.value=''; send.disabled=true; turns++;
  const pending=msg('bot','<span class="dots"><i></i><i></i><i></i></span>');
  const body=pending.querySelector('.bub');
  const t0=performance.now();
  try{
    const res=await fetch('/api/chat',{method:'POST',headers:{'content-type':'application/json'},
      body:JSON.stringify({message:text,sessionId:sessionId})});
    const r=await res.json();
    if(r.error) throw new Error(r.error);
    sessionId=r.sessionId||sessionId;
    if(r.is_fallback) pending.classList.add('refused');
    draw(r);
    document.getElementById('rt').textContent=Math.round(performance.now()-t0)+' ms round trip';
    await reveal(body,r.answer);
    if((r.citations||[]).length){
      const c=document.createElement('div'); c.className='cites';
      c.textContent='Sources: '+r.citations.join(', '); body.appendChild(c);
    }
    rate(body,r); clarify(body,r);
  }catch(e){
    pending.classList.add('refused');
    body.textContent='That request did not reach the model. Check that serve.py is still running.';
  }finally{ send.disabled=false; input.focus(); log.scrollTop=log.scrollHeight; }
}

form.addEventListener('submit',e=>{e.preventDefault(); ask(input.value);});

fetch('/api/health').then(r=>r.json()).then(h=>{
  document.getElementById('conn').textContent=h.ok?'live · '+h.region:'lambda unreachable';
  if(!h.ok) document.getElementById('live').classList.add('off');
}).catch(()=>{document.getElementById('conn').textContent='lambda unreachable';
              document.getElementById('live').classList.add('off');});

msg('bot','Ask me about orders, refunds, payments or your account. '+
          'Every answer shows how the model reached it in the panel on the right.');
input.focus();
</script>
</body>
</html>
"""


def invoke(payload: dict) -> dict:
    import boto3

    client = boto3.client("lambda", region_name=REGION)
    resp = client.invoke(
        FunctionName=FUNCTION,
        Payload=json.dumps({
            "requestContext": {"http": {"method": "POST"}},
            "body": json.dumps(payload),
        }).encode(),
    )
    body = json.loads(resp["Payload"].read())
    # The handler returns an API-Gateway-shaped response; unwrap it.
    return json.loads(body["body"]) if "body" in body else body


def record_feedback(payload: dict) -> dict:
    """Write the thumbs rating to DynamoDB.

    Written from the proxy rather than the Lambda because the rating arrives
    after the answer has already been returned -- a second Lambda round trip
    would buy nothing. The item shares the session partition key so a rating
    sits beside the turn it refers to.
    """
    import boto3
    from datetime import datetime, timezone

    table = boto3.resource("dynamodb", region_name=REGION).Table(
        os.environ.get("INTERACTIONS_TABLE", "faq-chatbot-interactions"))
    now = datetime.now(timezone.utc)
    table.put_item(Item={
        "pk": f"SESSION#{payload.get('sessionId') or 'web'}",
        "sk": f"RATING#{now.isoformat()}#{payload.get('request_id', '')}",
        "gsi1pk": f"RATING#{payload.get('intent') or 'UNKNOWN'}",
        "gsi1sk": now.isoformat(),
        "date": now.date().isoformat(),
        "request_id": payload.get("request_id"),
        "query": payload.get("query"),
        "intent": payload.get("intent"),
        "helpful": int(payload.get("helpful", 0)),
        "ttl": int(now.timestamp()) + 90 * 86400,
    })
    return {"ok": True}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/health":
            try:
                invoke({"message": "ping"})
                ok = {"ok": True, "function": FUNCTION, "region": REGION}
            except Exception as exc:
                ok = {"ok": False, "error": str(exc)[:200]}
            self._send(200, json.dumps(ok).encode(), "application/json")
        elif self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path not in ("/api/chat", "/api/feedback"):
            return self._send(404, b"not found", "text/plain")
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            started = time.perf_counter()
            if self.path == "/api/feedback":
                result = record_feedback(payload)
            else:
                result = invoke(payload)
            result["proxy_ms"] = round((time.perf_counter() - started) * 1000, 1)
            self._send(200, json.dumps(result).encode(), "application/json")
        except Exception as exc:
            self._send(502, json.dumps({"error": str(exc)[:300]}).encode(), "application/json")

    def log_message(self, fmt, *args):
        if "/api/chat" in (args[0] if args else ""):
            sys.stderr.write("  %s\n" % (fmt % args))


if __name__ == "__main__":
    print(f"Serving http://localhost:{PORT}  ->  lambda:{FUNCTION} ({REGION})")
    print("Ctrl+C to stop.")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

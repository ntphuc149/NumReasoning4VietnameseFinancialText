"use strict";
const $ = (s,r)=> (r||document).querySelector(s);
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const wait = ms => new Promise(r => setTimeout(r, REDUCED ? Math.min(ms,60) : ms));

/* Màn hình mở đầu chỉ viết một lần, trong index.html. Chụp lại nguyên trạng ở
   đây để nút "Cuộc trò chuyện mới" dựng lại đúng nó — khỏi phải giữ bản sao
   trong JS rồi hai bên lệch nhau. */
const INITIAL_COL = $("#col").innerHTML;

/* ------------------------------------------------------------- trạng thái */
const state = { pre:"", post:"", table:null, exId:null, running:false };
let draft = null;
let LIVE = false;                 // có backend thật hay không
let MODEL = "DeepSeek-V4-Flash";

/* =========================================================== khung soạn == */
const q = $("#query"), sendBtn = $("#send"), pop = $("#pop"), plus = $("#plus");

function autosize(){ q.style.height="auto"; q.style.height = Math.min(q.scrollHeight,180)+"px"; }
function syncSend(){
  const needKey = LIVE && BYOK && !hasCreds();
  sendBtn.disabled = state.running || q.value.trim()==="" || needKey;
  sendBtn.title = needKey ? "Nhập khoá API trước khi hỏi" : "Gửi";
  q.placeholder = needKey ? "Nhập khoá API ở đầu trang trước…" : "Nhập câu hỏi về số liệu…";
}
q.addEventListener("input", ()=>{ autosize(); syncSend(); state.exId=null; });
q.addEventListener("keydown", e=>{
  if(e.key==="Enter" && !e.shiftKey){ e.preventDefault(); if(!sendBtn.disabled) submit(); }
});

/* popover */
function setPop(open){
  pop.hidden = !open;
  plus.setAttribute("aria-expanded", String(open));
  if(open) pop.querySelector("button").focus();
}
plus.addEventListener("click", e=>{ e.stopPropagation(); setPop(pop.hidden); });
document.addEventListener("click", e=>{ if(!pop.hidden && !pop.contains(e.target)) setPop(false); });
document.addEventListener("keydown", e=>{ if(e.key==="Escape" && !pop.hidden){ setPop(false); plus.focus(); } });
pop.addEventListener("click", e=>{
  const b = e.target.closest("button[data-open]");
  if(b){ setPop(false); b.dataset.open==="ctx" ? openCtx() : openTbl(); }
});
$("#clear-all").addEventListener("click", ()=>{
  state.pre=""; state.post=""; state.table=null; renderSlips(); setPop(false);
});

/* phiếu đính kèm */
function ctxSummary(){
  const n = (state.pre?1:0) + (state.post?1:0);
  return n + " đoạn · " + (state.pre+state.post).length.toLocaleString("vi-VN") + " ký tự";
}
function renderSlips(){
  const slips = $("#slips");
  slips.innerHTML = "";
  const mk = (label,text,onEdit,onDel) => {
    const el = document.createElement("div");
    el.className = "slip";
    el.innerHTML = '<span class="k">'+label+'</span><span class="v">'+esc(text)+'</span>'+
      '<button type="button" data-a="edit" title="Sửa" aria-label="Sửa '+label+'"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20h4L19 9a2.1 2.1 0 0 0-3-3L5 17v3z"/></svg></button>'+
      '<button type="button" class="del" data-a="del" title="Xoá" aria-label="Xoá '+label+'"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg></button>';
    el.querySelector('[data-a="edit"]').addEventListener("click", onEdit);
    el.querySelector('[data-a="del"]').addEventListener("click", onDel);
    slips.appendChild(el);
  };
  if(state.pre || state.post)
    mk("Ngữ cảnh", ctxSummary(), openCtx, ()=>{ state.pre=""; state.post=""; renderSlips(); });
  if(state.table)
    mk("Bảng", state.table.length+" hàng × "+state.table[0].length+" cột", openTbl,
       ()=>{ state.table=null; renderSlips(); });
  $("#st-ctx").textContent = (state.pre||state.post) ? "đã có" : "";
  $("#st-tbl").textContent = state.table ? "đã có" : "";
}

/* =============================================== cửa sổ nhập ngữ cảnh == */
const dlgCtx = $("#dlg-ctx");
function openCtx(){
  $("#pre").value = state.pre; $("#post").value = state.post;
  updCtxCount(); dlgCtx.showModal(); $("#pre").focus();
}
function updCtxCount(){
  const n = ($("#pre").value + $("#post").value).length;
  $("#ctx-count").textContent = n.toLocaleString("vi-VN") + " ký tự";
}
$("#pre").addEventListener("input", updCtxCount);
$("#post").addEventListener("input", updCtxCount);
$("#form-ctx").addEventListener("submit", e=>{
  if(e.submitter && e.submitter.value === "save"){
    state.pre = $("#pre").value.trim();
    state.post = $("#post").value.trim();
    renderSlips();
  }
});

/* ================================================== cửa sổ nhập bảng == */
const dlgTbl = $("#dlg-tbl");
function blank(r,c){ return Array.from({length:r},()=>Array.from({length:c},()=>"")); }
function openTbl(){
  draft = state.table ? state.table.map(r=>r.slice()) : blank(4,3);
  $("#paste-zone").hidden = true;
  renderEditor(); dlgTbl.showModal();
  const first = $("#editor .cell"); if(first) first.focus();
}
function renderEditor(){
  const t = $("#editor");
  const cols = draft[0].length;
  let h = '<thead><tr><th class="gut-r gut-c corner"></th>';
  for(let c=0;c<cols;c++){
    h += '<th class="gut-c"><button type="button" data-delc="'+c+'" title="Xoá cột '+(c+1)+'" aria-label="Xoá cột '+(c+1)+'">×</button></th>';
  }
  h += '</tr></thead><tbody>';
  draft.forEach((row,r)=>{
    h += '<tr class="'+(r===0?"hdr":"")+'"><td class="gut-r"><button type="button" data-delr="'+r+'" title="Xoá hàng '+(r+1)+'" aria-label="Xoá hàng '+(r+1)+'">×</button></td>';
    row.forEach((cell,c)=>{
      h += '<td><input class="cell" data-r="'+r+'" data-c="'+c+'" value="'+esc(cell)+'" '+
           'placeholder="'+(r===0?"tiêu đề":"")+'" aria-label="Hàng '+(r+1)+' cột '+(c+1)+'"></td>';
    });
    h += '</tr>';
  });
  t.innerHTML = h + '</tbody>';
  $("#tbl-size").textContent = draft.length + " × " + cols;
  countCells();
}
function countCells(){
  $("#tbl-count").textContent = draft.flat().filter(v=>v.trim()!=="").length + " ô có dữ liệu";
}
$("#editor").addEventListener("input", e=>{
  const el = e.target.closest(".cell"); if(!el) return;
  draft[+el.dataset.r][+el.dataset.c] = el.value;
  countCells();
});
$("#editor").addEventListener("click", e=>{
  const dr = e.target.closest("[data-delr]"), dc = e.target.closest("[data-delc]");
  if(dr && draft.length>1){ draft.splice(+dr.dataset.delr,1); renderEditor(); }
  if(dc && draft[0].length>1){ draft.forEach(r=>r.splice(+dc.dataset.delc,1)); renderEditor(); }
});
$("#editor").addEventListener("paste", e=>{
  const el = e.target.closest(".cell"); if(!el) return;
  const txt = (e.clipboardData||window.clipboardData).getData("text");
  if(!/[\t\n]/.test(txt) && !/\|[^|]*\|/.test(txt)) return;
  e.preventDefault();
  spread(txt, +el.dataset.r, +el.dataset.c);
});

/* ------------------------------------------------- đọc bảng từ văn bản -- */
/** Dòng kẻ ngang của bảng Markdown: `|---|:---:|---:|` */
function isMdRule(line){
  return /^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$/.test(line);
}
function splitMdRow(line){
  let s = line.trim();
  if(s.startsWith("|")) s = s.slice(1);
  if(s.endsWith("|"))   s = s.slice(0, -1);
  return s.split(/(?<!\\)\|/).map(c => c.trim().replace(/\\\|/g, "|"));
}

/** Nhận Markdown, TSV hoặc CSV — tự đoán, trả về {rows, kind}. */
function parseTable(txt){
  const lines = txt.replace(/\r/g, "").split("\n").map(l => l.trim()).filter(Boolean);
  if(!lines.length) return { rows: [], kind: "" };

  const piped = lines.filter(l => l.includes("|")).length;
  if(piped >= lines.length - 1 && piped >= 1){
    const rows = lines.filter(l => !isMdRule(l)).map(splitMdRow);
    if(rows.length) return { rows: pad(rows), kind: "Markdown" };
  }
  if(lines.some(l => l.includes("\t")))
    return { rows: pad(lines.map(l => l.split("\t").map(c=>c.trim()))), kind: "TSV" };
  return { rows: pad(lines.map(l => l.split(",").map(c=>c.trim()))), kind: "CSV" };
}
/** Hàng ngắn thì đệm cho bằng hàng dài nhất. */
function pad(rows){
  const w = Math.max(...rows.map(r => r.length));
  return rows.map(r => { const c = r.slice(); while(c.length < w) c.push(""); return c; });
}

function spread(txt, r0, c0){
  const { rows } = parseTable(txt);
  if(!rows.length) return 0;
  const needR = r0 + rows.length, needC = c0 + Math.max(...rows.map(r=>r.length));
  while(draft.length < needR) draft.push(Array.from({length:draft[0].length},()=>""));
  if(needC > draft[0].length) draft.forEach(r=>{ while(r.length<needC) r.push(""); });
  rows.forEach((row,i)=> row.forEach((v,j)=>{ draft[r0+i][c0+j] = v.trim(); }));
  renderEditor();
  return rows.length;
}
$("#add-row").addEventListener("click", ()=>{ draft.push(Array.from({length:draft[0].length},()=>"")); renderEditor(); });
$("#add-col").addEventListener("click", ()=>{ draft.forEach(r=>r.push("")); renderEditor(); });
$("#toggle-paste").addEventListener("click", ()=>{
  const z = $("#paste-zone"); z.hidden = !z.hidden; if(!z.hidden) $("#paste-in").focus();
});
$("#do-paste").addEventListener("click", ()=>{
  const txt = $("#paste-in").value.trim();
  const msg = $("#paste-msg");
  if(!txt){ msg.textContent = "Chưa dán gì vào ô trên."; return; }

  const { rows, kind } = parseTable(txt);
  if(!rows.length){
    msg.textContent = "Không đọc được bảng. Cần Markdown (có dấu |), TSV hoặc CSV.";
    return;
  }
  draft = blank(1,1);
  spread(txt, 0, 0);
  msg.textContent = "";
  $("#paste-in").value = "";
  $("#paste-zone").hidden = true;
  $("#tbl-size").textContent = draft.length + " × " + draft[0].length + " · " + kind;
});
$("#form-tbl").addEventListener("submit", e=>{
  if(e.submitter && e.submitter.value === "save"){
    const rows = draft.filter(r => r.some(v => v.trim() !== ""));
    state.table = rows.length ? rows.map(r=>r.map(v=>v.trim())) : null;
    renderSlips();
  }
});

/* ================================================================ ví dụ == */
let SHOWN = EXAMPLES;            // danh sách ví dụ đang hiển thị

/** Rút ví dụ mới từ tập train qua backend. Không có backend thì giữ bản kèm sẵn. */
async function loadExamples(k){
  try {
    const res = await fetch("/api/examples?k=" + (k || 3), { cache:"no-store" });
    if(!res.ok) return false;
    const data = await res.json();
    if(!data.items || !data.items.length) return false;
    const src = $("#ex-src");
    if(src) src.textContent = "Ví dụ rút từ " + data.source;
    renderExamples(data.items);
    return true;
  } catch { return false; }
}

document.addEventListener("click", e=>{
  if(e.target.closest("#ex-more")) loadExamples(3);
});

function renderExamples(list){
  if(list) SHOWN = list;
  const g = $("#ex-grid"); if(!g) return;
  g.innerHTML = "";
  SHOWN.forEach(ex=>{
    const b = document.createElement("button");
    b.className = "ex"; b.type = "button";
    b.innerHTML = '<q>'+esc(ex.query)+'</q><div class="meta">'+
      '<span class="chip">Ngữ cảnh</span>'+
      '<span class="chip">Bảng '+ex.table.length+'×'+ex.table[0].length+'</span>'+
      '<span class="src">'+esc(ex.id)+'</span></div>';
    b.addEventListener("click", ()=>{
      state.pre = ex.pre; state.post = ex.post;
      state.table = ex.table.map(r=>r.slice());
      state.exId = ex.id;
      q.value = ex.query; autosize(); syncSend(); renderSlips(); q.focus();
    });
    g.appendChild(b);
  });
}

/* ================================================== hiển thị lượt người == */
function tableHTML(rows, hits){
  const hit = new Set((hits||[]).map(h=>h[0]+":"+h[1]));
  let h = '<div class="tbl-wrap"><table class="fin"><thead><tr>';
  rows[0].forEach(c => h += '<th>'+esc(c)+'</th>');
  h += '</tr></thead><tbody>';
  rows.slice(1).forEach((row,r)=>{
    h += '<tr>';
    row.forEach((c,ci)=> h += '<td data-rc="'+(r+1)+':'+ci+'" class="'+
      (hit.has((r+1)+":"+ci)?"hit":"")+'">'+esc(c)+'</td>');
    h += '</tr>';
  });
  return h + '</tbody></table></div>';
}
function paras(txt){ return txt.split(/\n+/).filter(Boolean).map(p=>'<p>'+esc(p)+'</p>').join(""); }
function caret(){
  return '<svg class="caret" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 5l7 7-7 7"/></svg>';
}
function renderUserTurn(p){
  const el = document.createElement("section");
  el.className = "turn-user reveal";
  let att = "";
  if(p.pre || p.post){
    const n = (p.pre?1:0)+(p.post?1:0);
    att += '<details class="att"><summary>'+caret()+'Ngữ cảnh · '+n+' đoạn · '+
      (p.pre+p.post).length.toLocaleString("vi-VN")+' ký tự</summary><div class="body">'+
      (p.pre ? '<span class="lbl tag">Trước bảng</span>'+paras(p.pre) : "")+
      (p.post ? '<span class="lbl tag" style="margin-top:12px">Sau bảng</span>'+paras(p.post) : "")+
      '</div></details>';
  }
  if(p.table){
    att += '<details class="att"><summary>'+caret()+'Bảng · '+p.table.length+' hàng × '+
      p.table[0].length+' cột</summary><div class="body" style="padding:13px">'+
      tableHTML(p.table, p.hits)+'</div></details>';
  }
  el.innerHTML = '<div class="bubble"><div class="q">'+esc(p.query)+'</div>'+att+'</div>';
  $("#col").appendChild(el);
  return el;
}

/* ==================================================== khung hiển thị trace */
const NODES = [
  { n:"Tách câu hỏi con", sub:"Chỉ định vị số liệu — không so sánh, không tính toán. (§4.1)" },
  { n:"Trả lời câu hỏi con", sub:"Mỗi câu hỏi con là một lời gọi độc lập, chạy song song. (§4.2)" },
  { n:"Lập kế hoạch", sub:"Lấy mẫu n kế hoạch ở nhiệt độ 0.6 thay vì giải mã tham lam một đường. (§4.3)" },
  { n:"Bỏ phiếu & thực thi", sub:"Chuẩn hoá, gom cụm, lấy cụm lớn nhất; hoà thì chọn kế hoạch ít bước hơn. (§4.4)" }
];
function toBottom(){ const t=$("#thread"); t.scrollTop = t.scrollHeight; }

class TraceView {
  constructor(userTurn){
    this.userTurn = userTurn;
    this.stages = [];
    this.answerRows = [];
    this.plansBox = null;
    this.el = document.createElement("section");
    this.el.className = "turn-bot reveal";
    this.el.innerHTML = '<div class="run-head"><span class="dot busy"></span>'+
      '<span class="t">Đang chạy đồ thị 4 tác tử…</span><span class="el">0,0 s</span></div>';
    $("#col").appendChild(this.el);
    this.t0 = performance.now();
    this.elp = this.el.querySelector(".el");
    this.tick = setInterval(()=>{
      this.elp.textContent = ((performance.now()-this.t0)/1000).toFixed(1).replace(".",",")+" s";
    }, 100);
    toBottom();
  }
  stageStart(i){
    const s = document.createElement("div");
    s.className = "stage busy";
    s.innerHTML = '<div class="rail"><div class="pip">'+(i+1)+'</div></div>'+
      '<div class="stage-body"><div class="stage-head"><h3>'+NODES[i].n+'</h3><span class="note"></span></div>'+
      '<div class="stage-sub">'+NODES[i].sub+'</div><div class="out"></div></div>';
    this.el.appendChild(s);
    this.stages[i] = s;
    toBottom();
    return s.querySelector(".out");
  }
  stageDone(i, note){
    const s = this.stages[i]; if(!s) return;
    s.classList.remove("busy"); s.classList.add("done");
    s.querySelector(".pip").innerHTML = '<svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 13l4 4L19 7"/></svg>';
    if(note) s.querySelector(".note").textContent = note;
  }
  out(i){ return this.stages[i].querySelector(".out"); }

  addSubquery(i, text){
    const d = document.createElement("div");
    d.className = "sq reveal";
    d.innerHTML = '<span class="tagx">sq_'+(i+1)+'</span><span class="txt">'+esc(text)+'</span><span></span>';
    this.out(0).appendChild(d); toBottom();
  }
  prepAnswers(questions){
    const box = this.out(1);
    this.answerRows = questions.map((text,i)=>{
      const d = document.createElement("div");
      d.className = "sq reveal";
      d.innerHTML = '<span class="tagx">v_'+(i+1)+'</span><span class="txt">'+esc(text)+
        '</span><span class="val wait">…</span>';
      box.appendChild(d);
      return d;
    });
    toBottom();
  }
  fillAnswer(i, short, full){
    const row = this.answerRows[i]; if(!row) return;
    const chip = row.querySelector(".val");
    if(chip) chip.outerHTML = '<span class="val">'+esc(short||"—")+'</span>';
    if(full) row.insertAdjacentHTML("beforeend", '<span class="from">'+esc(full)+'</span>');
    toBottom();
  }
  plannerPending(n){
    this.out(2).innerHTML = '<div class="plans reveal"><div class="plan-stat">'+
      '<span class="lbl">Đã lấy mẫu</span><b class="j-pc">0 / '+n+'</b>'+
      '<span class="lbl">Kế hoạch khác biệt</span><b class="j-pd">—</b></div>'+
      '<div class="bar-indet j-bar"></div>'+
      '<div class="votes j-vb" hidden></div>'+
      '<div class="vote-key j-vk">Đang lấy mẫu…</div></div>';
    this.plansBox = this.out(2).querySelector(".plans");
  }
  plansSampled(sampled, n){
    if(!this.plansBox) return;
    $(".j-pc", this.plansBox).textContent = sampled + " / " + n;
    const bar = $(".j-bar", this.plansBox); if(bar) bar.remove();
    toBottom();
  }
  showVotes(votes, total){
    if(!this.plansBox) return;
    $(".j-pd", this.plansBox).textContent = String(votes.length);
    const vb = $(".j-vb", this.plansBox);
    vb.hidden = false; vb.innerHTML = "";
    votes.forEach((v,i)=>{
      const sp = document.createElement("span");
      sp.style.flexGrow = String(v);
      sp.className = i===0 ? "win" : "";
      sp.textContent = v;
      vb.appendChild(sp);
    });
    const pct = total ? Math.round(votes[0]/total*100) : 0;
    $(".j-vk", this.plansBox).innerHTML = 'Cụm lớn nhất: <b class="mono">'+votes[0]+'/'+total+
      '</b> phiếu ('+pct+'%) — chọn làm phương trình cuối.';
    toBottom();
  }
  showHits(cells){
    if(!cells || !cells.length || !this.userTurn) return;
    cells.forEach(([r,c])=>{
      const td = this.userTurn.querySelector('[data-rc="'+r+':'+c+'"]');
      if(td) td.classList.add("hit");
    });
  }
  showFinal(f){
    let h = '<div class="prog">';
    (f.steps||[]).forEach(st=>{
      h += '<div class="row reveal"><span class="ref">'+esc(st.ref)+'</span>'+
        '<span class="expr">'+esc(st.expr)+'</span><span class="out">'+esc(st.out)+'</span></div>';
    });
    h += '</div>';
    const votes = f.votes || [];
    h += '<div class="answer reveal"><div class="top-row">'+
      '<div><div class="lbl" style="margin-bottom:6px">Đáp án</div>'+
      '<div class="val">'+esc(f.answer||"—")+'</div></div>'+
      (f.unit ? '<div class="unit">'+esc(f.unit)+'</div>' : '')+
      '<div class="conf">'+(votes[0]||0)+'/'+(f.plans||0)+' phiếu<br>'+
      (f.steps||[]).length+' bước</div></div>'+
      '<div class="foot"><code>'+esc(f.program||"—")+'</code>'+
      '<button class="copy" type="button" data-copy="'+esc(f.program||"")+'">Sao chép</button></div></div>';
    if(f.fallback){
      h += '<div class="warn" style="margin-top:12px">Không có kế hoạch nào chạy được — '+
           'đã rơi về nhắc trực tiếp (<span class="mono">'+esc(f.fallback)+'</span>).</div>';
    }
    this.out(3).innerHTML = h;
    toBottom();
  }
  finish(text){
    clearInterval(this.tick);
    this.el.querySelector(".dot").classList.remove("busy");
    this.el.querySelector(".t").textContent = text;
    this.elp.textContent = ((performance.now()-this.t0)/1000).toFixed(1).replace(".",",")+" s";
    toBottom();
  }
  fail(msg){
    clearInterval(this.tick);
    const dot = this.el.querySelector(".dot");
    dot.classList.remove("busy"); dot.classList.add("err");
    this.el.querySelector(".t").textContent = "Chạy lỗi";
    this.stages.forEach(s => s && s.classList.remove("busy"));
    const box = document.createElement("div");
    box.className = "warn bad";
    box.style.marginTop = "12px";
    box.textContent = msg;
    this.el.appendChild(box);
    toBottom();
  }
}

/* ------------------------------------------- gợi ý đơn vị cho đáp án ---- */
function unitHint(query, answer){
  const v = parseFloat(answer);
  if(isNaN(v)) return "";
  if(/tỷ lệ|phần trăm|%|tỷ trọng|tăng trưởng/i.test(query) && Math.abs(v) < 1)
    return "≈ " + (v*100).toFixed(1).replace(".",",") + "%";
  return "";
}

/* ============================================ chế độ thật (gọi backend) == */
async function runLive(view, payload){
  const res = await fetch("/api/run", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify(payload)
  });
  if(!res.ok || !res.body){
    // Máy chủ trả lỗi dạng {"error": "..."} — dùng câu đó nếu có.
    let why = "";
    try { why = (await res.json()).error || ""; } catch { /* không phải JSON */ }
    throw new Error(why || ("Máy chủ trả " + res.status + " " + res.statusText));
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "", nSamples = payload.n_samples, sampled = 0, errored = null;

  const handle = ev => {
    switch(ev.type){
      case "run_start":
        nSamples = ev.n_samples;
        break;
      case "stage":
        if(ev.status === "start"){
          view.stageStart(ev.i);
          if(ev.i === 2) view.plannerPending(nSamples);
        } else {
          view.stageDone(ev.i, ev.note + (ev.seconds ? " · " + ev.seconds + "s" : ""));
        }
        break;
      case "subqueries":
        ev.items.forEach((t,i)=> view.addSubquery(i, t));
        break;
      case "answers":
        view.prepAnswers(ev.items.map(x=>x.q));
        ev.items.forEach((x,i)=> view.fillAnswer(i, x.num || x.v, x.num ? x.v : ""));
        break;
      case "hits":
        view.showHits(ev.cells);
        break;
      case "plans":
        sampled = ev.sampled;
        view.plansSampled(sampled, nSamples);
        break;
      case "final":
        view.showVotes(ev.votes || [], ev.plans || sampled);
        view.showFinal({ ...ev, unit: unitHint(payload.query, ev.answer) });
        break;
      case "run_end": {
        const u = ev.usage || {};
        const bits = [];
        if(u.requests) bits.push(u.requests + " lời gọi");
        if(u.total_tokens) bits.push(u.total_tokens.toLocaleString("vi-VN") + " token");
        view.finish("Hoàn tất — " + MODEL + (bits.length ? " · " + bits.join(" · ") : ""));
        break;
      }
      case "error":
        errored = ev.message;
        break;
    }
  };

  for(;;){
    const { done, value } = await reader.read();
    if(done) break;
    buf += dec.decode(value, { stream:true });
    let i;
    while((i = buf.indexOf("\n\n")) >= 0){
      const chunk = buf.slice(0, i); buf = buf.slice(i+2);
      if(chunk.startsWith("data: ")){
        try { handle(JSON.parse(chunk.slice(6))); }
        catch(err){ console.error("SSE parse", err, chunk); }
      }
    }
  }
  if(errored) throw new Error(errored);
}

/* ==================================== chế độ mô phỏng (không có backend) = */
async function runMock(view, payload, trace){
  view.stageStart(0);
  await wait(700);
  for(let i=0;i<trace.subqueries.length;i++){
    view.addSubquery(i, trace.subqueries[i].q);
    await wait(170);
  }
  view.stageDone(0, "k = " + trace.subqueries.length);

  view.stageStart(1);
  view.prepAnswers(trace.subqueries.map(s=>s.q));
  for(let i=0;i<trace.subqueries.length;i++){
    await wait(320);
    view.fillAnswer(i, trace.subqueries[i].v, trace.subqueries[i].from);
  }
  view.showHits(trace.hits);
  view.stageDone(1, trace.subqueries.length + " lời gọi song song");

  view.stageStart(2);
  view.plannerPending(trace.plans);
  for(let i=1;i<=trace.plans;i++){
    view.plansSampled(i, trace.plans);
    await wait(REDUCED ? 0 : 50);
  }
  view.showVotes(trace.votes, trace.plans);
  view.stageDone(2, "n = " + trace.plans + ", T = 0.6");

  view.stageStart(3);
  await wait(600);
  view.showFinal(trace);
  view.stageDone(3, "đồng thuận");
  view.finish("Hoàn tất (mô phỏng) — 4 tác tử");
}

/* -- suy luận dự phòng cho đầu vào tự nhập khi không có backend --------- */
const numRe = /-?\d[\d.,]*/;
function parseNum(s){
  const m = String(s).match(numRe); if(!m) return null;
  const v = parseFloat(m[0].replace(/\.(?=\d{3}\b)/g,"").replace(/,/g,"."));
  return isNaN(v) ? null : v;
}
function fallbackTrace(payload){
  const tbl = payload.table, cells = [];
  if(tbl){
    for(let r=1;r<tbl.length;r++)
      for(let c=1;c<tbl[r].length;c++){
        const v = parseNum(tbl[r][c]);
        if(v!==null) cells.push({r,c,v,raw:tbl[r][c],row:tbl[r][0]||("hàng "+r),col:tbl[0][c]||("cột "+c)});
      }
  }
  const ql = payload.query.toLowerCase();
  const score = x => (x.row.toLowerCase().split(/\s+/).filter(w=>w.length>3 && ql.includes(w)).length*2)
                   + (x.col.toLowerCase().split(/\s+/).filter(w=>w.length>2 && ql.includes(w)).length);
  const pick = cells.slice().sort((a,b)=>score(b)-score(a)).slice(0,3);
  const subqueries = pick.length ? pick.map(x=>({
    q: 'Giá trị của "'+x.row+'" ở cột "'+x.col+'" là bao nhiêu?',
    v: x.raw, from: "bảng · hàng "+(x.r+1)+", cột "+(x.c+1)
  })) : [{ q:"Câu hỏi cần những số liệu nào trong tài liệu?", v:"—", from:"chưa có bảng để tra" }];

  let steps, program, answer, unit = "";
  const wantsRatio = /tỷ lệ|phần trăm|%|tỷ trọng|tăng trưởng/.test(ql);
  const wantsDiff  = /thay đổi|chênh lệch|tăng|giảm|nhiều hơn|ít hơn/.test(ql);
  if(pick.length>=2 && wantsRatio && wantsDiff){
    const d = pick[0].v - pick[1].v, r = pick[1].v ? d/pick[1].v : 0;
    steps = [ {ref:"#0", expr:"subtract("+pick[0].v+", "+pick[1].v+")", out:String(+d.toFixed(5))},
              {ref:"#1", expr:"divide(#0, "+pick[1].v+")", out:String(+r.toFixed(5))} ];
    program = steps[0].expr + ", " + steps[1].expr; answer = steps[1].out;
    unit = "≈ " + (r*100).toFixed(1).replace(".",",") + "%";
  } else if(pick.length>=2 && wantsRatio){
    const r = pick[1].v ? pick[0].v/pick[1].v : 0;
    steps = [ {ref:"#0", expr:"divide("+pick[0].v+", "+pick[1].v+")", out:String(+r.toFixed(5))} ];
    program = steps[0].expr; answer = steps[0].out;
    unit = "≈ " + (r*100).toFixed(1).replace(".",",") + "%";
  } else if(pick.length>=2){
    const d = pick[0].v - pick[1].v;
    steps = [ {ref:"#0", expr:"subtract("+pick[0].v+", "+pick[1].v+")", out:String(+d.toFixed(5))} ];
    program = steps[0].expr; answer = steps[0].out;
  } else {
    steps = [ {ref:"#0", expr:"—", out:"—"} ];
    program = "—"; answer = "—"; unit = "Cần bảng dữ liệu để suy luận";
  }
  const win = 9 + Math.floor(Math.random()*5), rest = 15 - win;
  const votes = rest > 3 ? [win, rest-2, 1, 1] : rest > 1 ? [win, rest-1, 1] : [win, rest].filter(v=>v>0);
  return { subqueries, hits: pick.map(x=>[x.r,x.c]), plans:15, distinct:votes.length,
           votes, steps, program, answer, unit };
}

/* ================================================================= gửi == */
async function submit(){
  const query = q.value.trim(); if(!query || state.running) return;
  state.running = true; syncSend();

  const nSel = $("#n-select");
  const payload = {
    query, pre:state.pre, post:state.post, table:state.table,
    n_samples: nSel ? +nSel.value : 15,
    model: (nSel && $("#model-select") ? $("#model-select").value : MODEL) || MODEL
  };
  // Chỉ gửi khoá khi máy chủ nói là nó không có khoá của riêng nó.
  if(BYOK && hasCreds()){ payload.api_key = CREDS.key; payload.base_url = CREDS.url; }

  const hello = $("#hello"), exs = $("#examples");
  if(hello) hello.remove();
  if(exs) exs.remove();

  // Ví dụ rút từ tập train không kèm trace dựng sẵn — chỉ bản đóng gói mới có.
  const ex = state.exId ? EXAMPLES.find(e => e.id === state.exId && e.trace) : null;
  const mockTrace = (ex && ex.query === query) ? ex.trace : fallbackTrace(payload);

  const userTurn = renderUserTurn({ ...payload, hits: LIVE ? null : mockTrace.hits });
  q.value = ""; autosize(); state.exId = null;
  toBottom();
  await wait(240);

  const view = new TraceView(userTurn);
  try {
    if(LIVE) await runLive(view, payload);
    else     await runMock(view, payload, mockTrace);
  } catch(err){
    view.fail(String(err.message || err));
  }
  state.running = false; syncSend();
}
sendBtn.addEventListener("click", submit);

/* sao chép chương trình */
document.addEventListener("click", e=>{
  const b = e.target.closest("[data-copy]"); if(!b) return;
  navigator.clipboard?.writeText(b.dataset.copy).then(()=>{
    const old = b.textContent; b.textContent = "Đã chép";
    setTimeout(()=>{ b.textContent = old; }, 1400);
  }).catch(()=>{});
});

/* ============================================================= làm mới == */
$("#reset").addEventListener("click", ()=>{
  if(state.running) return;
  state.pre=""; state.post=""; state.table=null; state.exId=null;
  q.value=""; autosize(); syncSend(); renderSlips();
  $("#col").innerHTML = INITIAL_COL;
  renderExamples();
  if(LIVE) loadExamples(3);
  q.focus();
});

/* ================================================= dò backend lúc khởi động */
/* =========================================== khoá của người xem (BYOK) == */
/* Chỉ nằm trong trình duyệt này. try/catch vì cửa sổ ẩn danh có thể ném lỗi. */
const KEY_STORE = "mpr.creds.v1";
let CREDS = { url:"", key:"", model:"" };
let BYOK = false;

function loadCreds(){
  try {
    const raw = localStorage.getItem(KEY_STORE);
    if(raw) CREDS = { url:"", key:"", model:"", ...JSON.parse(raw) };
  } catch { /* không đọc được thì coi như chưa có */ }
}
function saveCreds(){
  try { localStorage.setItem(KEY_STORE, JSON.stringify(CREDS)); } catch { /* bỏ qua */ }
}
function hasCreds(){ return !!(CREDS.url && CREDS.key); }

function syncKeyBtn(){
  const b = $("#key-btn");
  if(!b) return;
  b.hidden = !BYOK;
  b.className = "chip" + (BYOK && !hasCreds() ? " off" : "");
  b.textContent = hasCreds() ? "Khoá API ✓" : "Nhập khoá API";
}
function openKey(){
  $("#key-url").value = CREDS.url;
  $("#key-val").value = CREDS.key;
  $("#key-model").value = CREDS.model;
  $("#key-msg").textContent = "";
  $("#dlg-key").showModal();
  $("#key-url").focus();
}
$("#key-btn").addEventListener("click", openKey);
$("#key-clear").addEventListener("click", ()=>{
  CREDS = { url:"", key:"", model:"" };
  try { localStorage.removeItem(KEY_STORE); } catch { /* bỏ qua */ }
  $("#key-url").value = ""; $("#key-val").value = ""; $("#key-model").value = "";
  $("#key-msg").textContent = "Đã xoá khoá khỏi trình duyệt.";
  syncKeyBtn(); syncSend();
});
$("#form-key").addEventListener("submit", e=>{
  if(!e.submitter || e.submitter.value !== "save") return;
  CREDS = {
    url: $("#key-url").value.trim(),
    key: $("#key-val").value.trim(),
    model: $("#key-model").value.trim(),
  };
  saveCreds();
  if(CREDS.model && !MODEL_LIST.includes(CREDS.model)) MODEL_LIST.push(CREDS.model);
  fillModels(MODEL_LIST, CREDS.model || MODEL);
  syncKeyBtn(); syncSend();
});

let MODEL_LIST = [];

function fillModels(models, current){
  const sel = $("#model-select"); if(!sel) return;
  sel.innerHTML = "";
  (models && models.length ? models : [current]).forEach(m=>{
    const o = document.createElement("option");
    o.value = m; o.textContent = m; o.selected = (m === current);
    sel.appendChild(o);
  });
  sel.onchange = ()=>{ MODEL = sel.value; };
}

async function probe(){
  const chip = $("#status"), label = $("#status-t");
  const note = $("#mode-note"), foot = $("#tray-foot");

  // Chạy thật thì không nói gì — chỉ còn chấm xanh. Có trục trặc mới lên tiếng.
  const quiet = () => { label.textContent = ""; note.textContent = ""; foot.hidden = true; };
  const speak = (short, long) => {
    label.textContent = short; note.textContent = long; foot.hidden = false;
  };

  try {
    const res = await fetch("/api/health", { cache:"no-store" });
    if(!res.ok) throw new Error("health " + res.status);
    const info = await res.json();
    LIVE = !!info.ok;
    BYOK = !!info.byok;
    MODEL = info.model || MODEL;
    MODEL_LIST = (info.models || [MODEL]).slice();
    if(CREDS.model && !MODEL_LIST.includes(CREDS.model)) MODEL_LIST.push(CREDS.model);
    fillModels(MODEL_LIST, CREDS.model || MODEL);
    syncKeyBtn();
    if(LIVE){
      chip.className = "chip on";
      chip.title = "Đang gọi mô hình thật";
      quiet();
      loadExamples(3);
    } else {
      chip.className = "chip off";
      chip.title = "Máy chủ chạy nhưng chưa cấu hình được khoá";
      speak("thiếu khoá", "Thiếu API_KEY/BASE_URL — đang dùng bản mô phỏng");
    }
  } catch {
    LIVE = false; BYOK = false;
    chip.className = "chip off";
    chip.title = "Không thấy backend";
    fillModels(MODEL_LIST.length ? MODEL_LIST : null, MODEL);
    syncKeyBtn();
    speak("mô phỏng", "Chưa có backend — chạy `python demo/server.py` để gọi thật");
  }
}

loadCreds();
renderExamples();
renderSlips();
autosize();
probe();

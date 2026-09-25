"""Web-based click-calibration tool for scene geometry (CAMERA-LEVEL, shared).

All sample videos are treated as ONE fixed camera: they share the same
resolution (3840x2160) and scene geometry. There is a single camera-level
`scene_config.json` that applies to every video — including hidden test
videos — with NO per-video configuration. The Video selector in this tool is
for VIEWING/CHECKING the same shared geometry on different samples only; every
clicks/promote edit the ONE shared config.

Usage (from the package dir, any browser on the same machine):
    python debug\\calibrate_scene.py [--port 8090] [--width 1200]
    ->  http://127.0.0.1:8090

Workflow
  1. Pick a sample video (view only) + frame index, "Load".
  2. Choose a feature type, click on the canvas to place points.
     Road/lane/crosswalk/intersection/exclusion/u_turn are polygons;
     stop/solid are 2-point lines; traffic light = outline -> saved as bbox ROI.
  3. "Close" the shape (lane asks for id + direction).
  4. "Save draft" writes debug/scene_calib.json (and prints full-res coords
     to the server console so they can be pasted around).
  5. "Promote to scene_config.json" writes the draft into the SHARED
     top-level features of the single scene_config.json.

Coordinates are stored in full 3840x2160-equivalent pixels (converted from
canvas clicks via the loaded frame's real size); src/geometry.py scales them
to the actual frame resolution at runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2

VIDEO_DIR = r"C:\Projects\Traffic Computer Vision\video"
VIDEOS = ["C3896.MP4", "C3897.MP4", "C3902.MP4", "C3905.MP4",
          "C3905_preview.mp4"]
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(PACKAGE_DIR, "scene_config.json")
DRAFT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_calib.json")

sys.path.insert(0, PACKAGE_DIR)
from src.geometry import Geometry  # noqa: E402

CANVAS_W = 1200

TYPE_KEYS = {
    "road": "road_polygon",
    "lane": "lanes",
    "crosswalk": "crosswalks",
    "exclusion": "exclusion_regions",
    "intersection": "intersection_zones",
    "u_turn": "u_turn_zones",
    "traffic": "traffic_light_rois",
    "stop": "stop_lines",
    "solid": "solid_lines",
}


def read_frame(video_path: str, index: int):
    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        idx = max(0, min(int(index), total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:  # H.264 seek hiccup -> scan forward from 0 up to +90
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for i in range(idx + 90):
                ok, fr = cap.read()
                if ok and i >= idx:
                    break
        return fr if ok else None, total, W, H
    finally:
        cap.release()


def load_cfg():
    if not os.path.exists(CONFIG):
        return {}
    with open(CONFIG, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cfg(cfg: dict):
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")


def empty_features() -> dict:
    return {
        "road_polygon": {"enabled": True, "confidence": "calibrated",
                         "points": [], "note": ""},
        "lanes": [],
        "crosswalks": [],
        "exclusion_regions": [],
        "intersection_zones": [],
        "u_turn_zones": [],
        "traffic_light_rois": [],
        "stop_lines": [],
        "solid_lines": [],
    }


def reference_overlay() -> dict:
    """Camera-level (shared) geometry for the current file state."""
    cfg = load_cfg()
    if not cfg:
        return {}
    g = Geometry(cfg, frame_w=3840, frame_h=2160)
    return {
        "road_polygon": g.road_polygon,
        "lanes": [{"lane_id": l["lane_id"], "polygon": l["polygon"]} for l in g.lanes],
        "crosswalks": g.crosswalks,
        "exclusion_regions": g.exclusion_regions,
        "intersection_zones": g.intersections,
        "u_turn_zones": g.u_turn_zones,
        "traffic_light_rois": g.traffic_light_rois,
        "stop_lines": g.stop_lines,
        "solid_lines": g.solid_lines,
    }


def promote(features: dict) -> str:
    """Write draft into the SHARED top-level feature keys of the single config."""
    cfg = load_cfg()
    block = empty_features()
    feat = features or {}
    for k, default in block.items():
        val = feat.get(k)
        block[k] = val if val is not None else default
    for k in list(block):
        if k != "road_polygon" and isinstance(block[k], list) and not block[k]:
            block.pop(k)
    cfg.update(block)
    save_cfg(cfg)
    n = sum(1 for v in block.values()
            if isinstance(v, list) and v) + (1 if block["road_polygon"]["points"] else 0)
    return f"promoted {n} feature(s) into SHARED camera-level {CONFIG}"


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Scene geometry calibrator</title>
<style>
 body { font: 14px/1.5 Segoe UI, sans-serif; background:#111; color:#ddd; margin:12px; }
 h1 { font-size:18px; }
 .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:6px 0; }
 label { font-size:12px; color:#aaa; }
 select,input,button { font-size:13px; padding:4px 6px; background:#222; color:#eee;
   border:1px solid #444; border-radius:4px; }
 button { cursor:pointer; } button:hover { border-color:#888; }
 button.hl { color:#111; background:#ffd74f; border-color:#ffd74f; font-weight:bold; }
 canvas { background:#000; border:1px solid #555; margin-top:8px; max-width:100%; }
 #status { color:#9fdf7f; font-size:12px; white-space:pre-wrap; }
 #coords { font-family:Consolas,monospace; font-size:11px; color:#8cd; white-space:pre;
   background:#1a1a1a; border:1px solid #333; padding:6px; height:150px; overflow:auto; }
 .legend span { display:inline-block; width:12px; height:12px; margin:0 4px 0 14px;
   vertical-align:-1px; border:1px solid rgba(255,255,255,.4); }
 .ref { stroke-dasharray: 6 5; }
</style>
</head>
<body>
<h1>Scene geometry calibrator &mdash; camera-level (shared)</h1>
<div class="row">
  <label>Video (view only) <select id="video"></select></label>
  <label>Frame <input id="frameIdx" type="number" min="0" value="0"></label>
  <button onclick="loadFrame()">Load</button>
  <span id="fpsInfo" style="font-size:12px;color:#888"></span>
</div>
<div class="row">
  <label>Type <select id="type">
    <option value="road">Road area</option>
    <option value="lane">Lane</option>
    <option value="crosswalk">Crosswalk</option>
    <option value="intersection">Intersection</option>
    <option value="exclusion">Exclusion / off-road</option>
    <option value="u_turn">U-turn zone</option>
    <option value="traffic">Traffic light ROI (2 pts)</option>
    <option value="stop">Stop line (2 pts)</option>
    <option value="solid">Solid line (2 pts)</option>
  </select></label>
  <label>Lane id <input id="laneId" size="8" placeholder="L1" disabled></label>
  <label>Dir&deg; <input id="laneDir" size="4" value="90" disabled></label>
  <label>Note <input id="note" size="22" placeholder="optional"></label>
</div>
<div class="row">
  <button onclick="undoPoint()">Undo pt</button>
  <button class="hl" onclick="closeShape()">Close shape</button>
  <button onclick="deleteLast()">Delete last of type</button>
  <button onclick="clearDraft()">Clear draft</button>
  <span style="flex:1"></span>
  <button onclick="saveDraft()">Save draft</button>
  <button class="hl" onclick="promote()">Promote to scene_config.json</button>
</div>
<div class="row" style="color:#999;font-size:12px">
  All four samples are ONE fixed camera (same resolution 3840&times;2160): the draft edits the
  single SHARED camera-level geometry, which applies to every video &mdash; hidden test videos included.
</div>
<div class="row legend" id="legend"></div>
<div class="row"><label><input type="checkbox" id="showRef" checked> active geometry reference (thin dashed)</label></div>
<canvas id="cv"></canvas>
<div class="row"><b>Full-res coordinates (draft)</b></div>
<pre id="coords"></pre>
<div id="status">ready</div>
<script>
"use strict";
const T = {
  road:{color:'#2ecc71',pts:'poly',label:'Road'},
  lane:{color:'#3498db',pts:'poly',label:'Lane'},
  crosswalk:{color:'#e67e22',pts:'poly',label:'Crosswalk'},
  intersection:{color:'#9b59b6',pts:'poly',label:'Intersection'},
  exclusion:{color:'#e74c3c',pts:'poly',label:'Exclusion'},
  u_turn:{color:'#f1c40f',pts:'poly',label:'U-turn'},
  traffic:{color:'#1abc9c',pts:'line',label:'Traffic'},
  stop:{color:'#7fe0c3',pts:'line',label:'Stop'},
  solid:{color:'#95a5a6',pts:'line',label:'Solid'},
};
const KEYS={road:'road_polygon',lane:'lanes',crosswalk:'crosswalks',exclusion:'exclusion_regions',
  intersection:'intersection_zones',u_turn:'u_turn_zones',traffic:'traffic_light_rois',
  stop:'stop_lines',solid:'solid_lines'};
const videoSel=document.getElementById('video');
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
let fullW=3840, fullH=2160, img=null, ref=null, feat=null, cur=[], video=null, frameIdx=0;

function emptyFeat(){const o={};for(const k of Object.values(KEYS))o[k]=[];return o;}
async function api(path){const r=await fetch(path);if(!r.ok)throw new Error(path+' '+r.status);
  return JSON.parse(await r.text());}
function setStatus(m){document.getElementById('status').textContent=m;}
function setCoords(c){document.getElementById('coords').textContent=c;}
function onImg(src){const i=new Image();i.onload=()=>{img=i;draw();};i.src=src;}

function buildLegend(){const el=document.getElementById('legend');el.innerHTML='';
  for(const k in T)el.innerHTML+='<span style="background:'+T[k].color+'"></span>'+T[k].label;}

['C3896.MP4','C3897.MP4','C3902.MP4','C3905.MP4','C3905_preview.mp4']
  .forEach(v=>videoSel.add(new Option(v,v)));

async function loadFrame(){
  video=videoSel.value; frameIdx=parseInt(document.getElementById('frameIdx').value)||0;
  document.getElementById('frameIdx').value=frameIdx;
  try{
    const st=await api('/state?v='+encodeURIComponent(video));
    feat=st.features||emptyFeat(); fullW=st.width||3840; fullH=st.height||2160;
    if(frameIdx===0 && st.total){frameIdx=Math.floor(st.total/2);
      document.getElementById('frameIdx').value=frameIdx;}
    ref=await api('/ref?v='+encodeURIComponent(video));
    onImg('/frame?v='+encodeURIComponent(video)+'&i='+frameIdx+'&w=1200');
    setStatus('loaded '+video+' frame '+frameIdx+' / '+st.total+'  ['+st.width+'x'+st.height+']');
  }catch(e){setStatus('error: '+e.message);}
}

cv.addEventListener('click',e=>{
  const t=document.getElementById('type').value; if(!t)return;
  const r=cv.getBoundingClientRect();
  cur.push([Math.round((e.clientX-r.left)*fullW/cv.width),
            Math.round((e.clientY-r.top)*fullH/cv.height)]);
  draw(); listCoords();
});
cv.addEventListener('contextmenu',e=>{e.preventDefault();undoPoint();});

function undoPoint(){cur.pop();draw();listCoords();}

function closeShape(){
  const t=document.getElementById('type').value; const key=KEYS[t];
  if(t==='lane'){document.getElementById('laneId').disabled=false;

    document.getElementById('laneDir').disabled=false;}
  if(cur.length===0)return;
  const common={enabled:true,confidence:'calibrated',note:document.getElementById('note').value};
  if(T[t].pts==='poly'){
    if(cur.length<3){setStatus('polygon needs >=3 points');return;}
    if(t==='road'){feat.road_polygon={points:cur.slice(),...common};}
    else if(t==='lane'){
      const id=document.getElementById('laneId').value||('L'+feat.lanes.length);
      const dir=parseFloat(document.getElementById('laneDir').value)||0;
      feat.lanes.push({lane_id:id,expected_direction:dir,polygon:cur.slice(),...common});
    }else{
      feat[key].push({polygon:cur.slice(),...common});
    }
  }else{
    if(cur.length<2){setStatus('line needs 2 points');return;}
    if(t==='stop'||t==='solid'){
      if(cur.length!==2){setStatus('stop/solid take EXACTLY 2 points');return;}
      feat[key].push({line:cur.slice(),...common});
    }else{ // traffic -> bbox line from the clicked outline
      const xs=cur.map(p=>p[0]), ys=cur.map(p=>p[1]);
      feat[key].push({line:[[Math.min(...xs),Math.min(...ys)],
                            [Math.max(...xs),Math.max(...ys)]],...common});
    }
  }
  cur=[]; draw(); listCoords(); setStatus('closed '+t);
}

function deleteLast(){
  const t=document.getElementById('type').value; const key=KEYS[t];
  if(t==='road'){feat.road_polygon={enabled:true,confidence:'calibrated',points:[],note:''};}
  else if(Array.isArray(feat[key]))feat[key].pop();
  else feat[key]=[];
  draw(); listCoords();
}

function clearDraft(){feat=emptyFeat();cur=[];draw();listCoords();setStatus('draft cleared');}

function draw(){
  if(!img)return;
  const w=1200, h=Math.round(img.height*w/img.width);
  cv.width=w; cv.height=h; ctx.clearRect(0,0,w,h); ctx.drawImage(img,0,0,w,h);
  const sx=w/fullW, sy=h/fullH;
  if(document.getElementById('showRef').checked && ref)drawRef(sx,sy);
  for(const k in KEYS){const key=KEYS[k];drawItems(feat?feat[key]:[],sx,sy,T[k].color,key==='road_polygon',
    k==='stop'?8:5,k==='lane');}
  if(cur.length>0){
    ctx.save(); ctx.fillStyle='#fff';
    ctx.moveTo(cur[0][0]*sx,cur[0][1]*sy);
    for(const p of cur.slice(1))ctx.lineTo(p[0]*sx,p[1]*sy);
    ctx.strokeStyle='#fff'; ctx.setLineDash([6,4]); ctx.lineWidth=2; ctx.stroke();
    ctx.restore();
  }
}

function drawRef(sx,sy){
  ctx.save(); ctx.strokeStyle='rgba(255,255,255,0.55)'; ctx.fillStyle='rgba(255,255,255,0.12)';
  ctx.lineWidth=1; ctx.setLineDash([6,5]);
  const poly=(p)=>{if(!p||p.length<3)return;ctx.beginPath();ctx.moveTo(p[0][0]*sx,p[0][1]*sy);
    for(const pt of p.slice(1))ctx.lineTo(pt[0]*sx,pt[1]*sy);ctx.closePath();ctx.fill();ctx.stroke();};
  poly(ref.road_polygon);
  for(const l of ref.lanes||[])poly(l.polygon);
  for(const p of ref.crosswalks||[])poly(p);
  for(const p of ref.exclusion_regions||[])poly(p);
  for(const p of ref.intersection_zones||[])poly(p);
  for(const p of ref.u_turn_zones||[])poly(p);
  for(const l of ref.stop_lines||[])line(l); 
  for(const l of ref.solid_lines||[])line(l);
  for(const l of ref.traffic_light_rois||[])line(l);
  function line(l){ctx.beginPath();ctx.moveTo(l[0][0]*sx,l[0][1]*sy);
    ctx.lineTo(l[1][0]*sx,l[1][1]*sy);ctx.stroke();}
  ctx.restore();
}

function drawItems(items,sx,sy,color,isRoad,width,isLane){
  if(isRoad)items=[feat&&feat.road_polygon&&feat.road_polygon.points
    ?{polygon:feat.road_polygon.points}:null];
  for(const it of (items||[])){
    const pts=isRoad?(it&&it.polygon):(it?it.polygon||it.line:null);
    if(!pts||pts.length<2)continue;
    ctx.beginPath(); ctx.moveTo(pts[0][0]*sx,pts[0][1]*sy);
    for(const p of pts.slice(1))ctx.lineTo(p[0]*sx,p[1]*sy);
    ctx.strokeStyle=color; ctx.lineWidth=width; ctx.stroke();
    ctx.fillStyle=color+'55'; ctx.fill();
    ctx.fillStyle=color; ctx.font='bold 11px Segoe UI';
    const lbl=isRoad?'ROAD':(isLane?(it.lane_id||'L'):'');
    if(lbl)ctx.fillText(lbl,pts[0][0]*sx+4,pts[0][1]*sy-4);
  }
} 

function listCoords(){
  let s=''; const ROAD='road';
  if(feat&&feat.road_polygon&&feat.road_polygon.points&&feat.road_polygon.points.length)
    s+='ROAD '+JSON.stringify(feat.road_polygon.points)+'\\n';
  for(const k in KEYS){const key=KEYS[k];if(k==='road')continue;
    const items=feat&&feat[key]?feat[key]:[];
    for(const it of items){const p=it.polygon||it.line;
      if(p)s+=k.toUpperCase()+' '+(it.lane_id?it.lane_id+' ':'')+JSON.stringify(p)+'\\n';}}
  if(cur.length)s+='CUR '+JSON.stringify(cur);
  document.getElementById('coords').textContent=s;
}

async function saveDraft(){
  try{
    const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({video,frame_index:frameIdx,features:feat})});
    const j=await r.json(); setStatus(j.msg||'saved');
  }catch(e){setStatus('error: '+e.message);}
}
async function promote(){
  try{
    const r=await fetch('/promote',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({video,features:feat})});
    const j=await r.json(); setStatus(j.msg||'promoted');
    ref=await api('/ref?v='+encodeURIComponent(video)); draw();
  }catch(e){setStatus('error: '+e.message);}
}

document.getElementById('type').onchange=()=>{
  const t=document.getElementById('type').value;
  document.getElementById('laneId').disabled=t!=='lane';
  document.getElementById('laneDir').disabled=t!=='lane';
};
document.getElementById('showRef').onchange=draw;
buildLegend(); video=videoSel.value; feat=emptyFeat(); loadFrame();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            return self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
        if u.path == "/state":
            video = q.get("v", [""])[0]
            path = os.path.join(VIDEO_DIR, video)
            _, total, W, H = read_frame(path, 0)
            cfg = load_cfg()
            feats = empty_features()
            for k in feats:
                if k in cfg:
                    feats[k] = cfg[k]
            body = json.dumps({"video": video, "features": feats,
                               "width": W, "height": H, "total": total}).encode()
            return self._send(200, body)
        if u.path == "/ref":
            return self._send(200, json.dumps(reference_overlay()).encode())
        if u.path == "/frame":
            video = q.get("v", [""])[0]
            idx = int(q.get("i", ["0"])[0])
            w = int(q.get("w", [str(CANVAS_W)])[0])
            fr, total, W, H = read_frame(os.path.join(VIDEO_DIR, video), idx)
            if fr is None:
                return self._send(404, b"frame read failed")
            h = int(fr.shape[0] * w / fr.shape[1])
            ok, buf = cv2.imencode(".jpg", cv2.resize(fr, (w, h)),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                return self._send(500, b"encode failed")
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("X-Total-Frames", str(total))
            self.send_header("Content-Length", str(len(buf)))
            self.end_headers()
            self.wfile.write(buf.tobytes())
            return

    def do_POST(self):
        u = urlparse(self.path)
        if u.path not in ("/save", "/promote"):
            return self._send(404, b"404")
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as ex:  # noqa: BLE001
            return self._send(400, json.dumps({"error": str(ex)}).encode())
        video = body.get("video", "")
        feats = body.get("features", {})
        if u.path == "/save":
            with open(DRAFT, "w", encoding="utf-8") as f:
                json.dump({"video": video, "frame_index": body.get("frame_index", 0),
                           "features": feats}, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"\nsaved draft for {video}:")
            print(json.dumps(feats, ensure_ascii=False, indent=1))
            msg = f"draft saved to {DRAFT}"
        else:
            msg = promote(feats)
            print(msg)
        return self._send(200, json.dumps({"ok": True, "msg": msg}).encode())


def main():
    global CANVAS_W
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--width", type=int, default=CANVAS_W)
    a = ap.parse_args()
    CANVAS_W = a.width
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Open http://127.0.0.1:{a.port}  (Ctrl+C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
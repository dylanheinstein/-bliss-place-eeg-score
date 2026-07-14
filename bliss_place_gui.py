#!/opt/homebrew/bin/python3
import tkinter as tk
from tkinter import font
import subprocess, threading, time, os, signal
from collections import deque

try:
    import redis as _redis; _redis_ok = True
except ImportError:
    _redis_ok = False

os.environ["TK_SILENCE_DEPRECATION"] = "1"

BG        = "#0a0a0f"
PANEL     = "#13131e"
BORDER    = "#22223a"
ACCENT    = "#7c6aff"
DELTA_COL = "#4a8fd4"
THETA_COL = "#3dbf9e"
ALPHA_COL = "#7c6aff"
BETA_COL  = "#d46ab0"
TEXT      = "#e8e8f0"
DIM       = "#4a4a6a"
GREEN     = "#4ecb7a"
RED       = "#e05c5c"
CHORD_COL = "#f0c060"

BAND_COLORS = {"delta":DELTA_COL,"theta":THETA_COL,"alpha":ALPHA_COL,"beta":BETA_COL}

SCRIPTS = {
    "buffer":    ("Buffer",    "~/Desktop/eegsynth-master/src/module/buffer/buffer.py",
                               "-i ~/Desktop/eegsynth-master/patches/myfirstpatch/buffer.ini"),
    "neuropawn": ("NeuroPawn", "~/Desktop/neuropawn_stream.py", ""),
    "spectral":  ("Spectral",  "~/Desktop/eegsynth-master/src/module/spectral/spectral.py",
                               "-i ~/Desktop/eegsynth-master/patches/myfirstpatch/spectral.ini"),
    "music":     ("Music",     "~/Desktop/brain_music.py", ""),
    "drums":     ("Drums",     "~/Desktop/drum_synth.py", ""),
    "plot":      ("Plot",      "~/Desktop/eegsynth-master/src/module/plotsignal/plotsignal.py",
                               "-i ~/Desktop/eegsynth-master/patches/myfirstpatch/plotsignal.ini"),
}

STATE_LABELS = {0:"deep-delta",1:"theta",2:"theta/alpha",3:"alpha",
                4:"alpha",5:"alpha/beta",6:"beta",7:"beta"}
CHORD_NAMES  = ["Cmaj","Dmin","Emin","Fmaj","Gmaj","Amin","Bdim","Cmaj+"]


class BandReader:
    def __init__(self):
        self.r=None; self.connected=False
        self.hist  = {b:deque(maxlen=30) for b in ("delta","theta","alpha","beta")}
        self.ema   = {b:0.5 for b in ("delta","theta","alpha","beta")}
        self.raw_v = {b:0.0 for b in ("delta","theta","alpha","beta")}
        self._try()
    def _try(self):
        if not _redis_ok: return
        try: self.r=_redis.Redis(); self.r.ping(); self.connected=True
        except: self.connected=False
    def read(self):
        if not self.connected: self._try(); return
        for band,key in [("delta","spectral.channel1.delta"),("theta","spectral.channel1.theta"),
                          ("alpha","spectral.channel1.alpha"),("beta","spectral.channel1.beta")]:
            try:
                v=self.r.get(key)
                if v:
                    v=float(v); self.raw_v[band]=v; self.hist[band].append(v)
                    lo,hi=min(self.hist[band]),max(self.hist[band])
                    n=(v-lo)/(hi-lo) if hi!=lo else 0.5
                    self.ema[band]=0.2*n+0.8*self.ema[band]
            except: pass
    def get_str(self,k):
        try: v=self.r.get(k); return v.decode() if v else "—"
        except: return "—"


def F(parent, bg, x, y, w, h):
    f = tk.Frame(parent, bg=bg, width=w, height=h)
    f.place(x=x, y=y); f.pack_propagate(False); return f

def L(parent, text, size=10, bold=False, mono=False, color=TEXT, anchor="w"):
    f = font.Font(family="Courier" if mono else "Helvetica Neue",
                  size=size, weight="bold" if bold else "normal")
    return tk.Label(parent, text=text, font=f, bg=parent["bg"], fg=color, anchor=anchor)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bliss Place — EEG Brain Music")
        self.configure(bg=BG); self.geometry("900x640"); self.resizable(False,False)
        self.reader=BandReader(); self.procs={}; self.volumes={}; self.n=0
        self._ui(); self._poll()

    def _ui(self):
        # ── header ──
        h = F(self, BG, 0, 0, 900, 52)
        L(h,"BLISS PLACE",18,bold=True,color=TEXT).place(x=24,y=10)
        L(h,"EEG Brain Music Controller",11,color=DIM).place(x=178,y=17)
        L(h,"Redis",9,color=DIM).place(x=824,y=8)
        self.rdot = tk.Label(h,text="●",font=font.Font(family="Helvetica Neue",size=14),
                             bg=BG,fg=RED); self.rdot.place(x=856,y=6)
        tk.Frame(self,bg=BORDER,height=1).place(x=0,y=52,width=900)

        # ── band meters ──
        mp = F(self, PANEL, 24, 62, 580, 228)
        L(mp,"BRAINWAVE BANDS",9,color=DIM).place(x=12,y=8)
        self._bars,self._bv,self._br = {},{},{}
        BW=360
        for i,band in enumerate(("delta","theta","alpha","beta")):
            y=30+i*42; col=BAND_COLORS[band]
            L(mp,band.upper(),10,mono=True,color=col).place(x=12,y=y)
            tk.Frame(mp,bg=BORDER,width=BW,height=16).place(x=74,y=y+1)
            bar=tk.Frame(mp,bg=col,width=0,height=16); bar.place(x=74,y=y+1)
            vl=L(mp,"0.00",10,mono=True,color=TEXT,anchor="e"); vl.place(x=442,y=y)
            rl=L(mp,"—",9,mono=True,color=DIM,anchor="w"); rl.place(x=492,y=y+1)
            self._bars[band]=(BW,bar); self._bv[band]=vl; self._br[band]=rl
        # score
        y=30+4*42
        L(mp,"SCORE",10,mono=True,color=ACCENT).place(x=12,y=y)
        tk.Frame(mp,bg=BORDER,width=BW,height=16).place(x=74,y=y+1)
        self._sbar=tk.Frame(mp,bg=ACCENT,width=0,height=16); self._sbar.place(x=74,y=y+1)
        self._sv=L(mp,"0.00",10,mono=True,color=TEXT,anchor="e"); self._sv.place(x=442,y=y)

        # ── chord panel ──
        cp=F(self,PANEL,24,300,580,110)
        L(cp,"CHORD",9,color=DIM).place(x=12,y=8)
        self._cl=L(cp,"—",28,bold=True,color=CHORD_COL); self._cl.place(x=10,y=26)
        L(cp,"STATE",9,color=DIM).place(x=190,y=8)
        self._sl=L(cp,"—",13,color=ALPHA_COL); self._sl.place(x=190,y=26)
        L(cp,"ITER",9,color=DIM).place(x=370,y=8)
        self._il=L(cp,"0000",13,mono=True,color=DIM); self._il.place(x=370,y=26)

        # ── readout panel ──
        rp=F(self,PANEL,24,402,580,78)
        L(rp,"LIVE READOUT",9,color=DIM).place(x=12,y=6)
        L(rp,"MELODY",10,mono=True,color=THETA_COL).place(x=12,y=26)
        self._ml=L(rp,"—",10,mono=True,color=TEXT); self._ml.place(x=90,y=26)
        L(rp,"GLIDE",10,mono=True,color=BETA_COL).place(x=12,y=48)
        self._gl=L(rp,"—",10,mono=True,color=TEXT); self._gl.place(x=90,y=48)

        # ── volume panel ──
        vp=F(self,PANEL,24,490,580,120)
        L(vp,"VOLUMES",9,color=DIM).place(x=12,y=6)
        for i,(lbl,key) in enumerate([("Chord","chord"),("Melody","melody"),
                                       ("Glide","glide"),("Drums","drums")]):
            x=20+i*140
            L(vp,lbl,9,color=DIM).place(x=x+28,y=24)
            var=tk.DoubleVar(value=0.7); self.volumes[key]=var
            tk.Scale(vp,variable=var,from_=0.0,to=1.0,resolution=0.01,
                     orient="vertical",bg=PANEL,fg=TEXT,troughcolor=BORDER,
                     highlightthickness=0,bd=0,length=72,showvalue=False,
                     command=lambda v,k=key:self._vol(k,v)).place(x=x+18,y=38)

        # ── modules panel ──
        mod=F(self,PANEL,618,62,258,548)
        L(mod,"MODULES",9,color=DIM).place(x=12,y=8)
        self._btns,self._dots={},{}
        for i,(key,(label,s,a)) in enumerate(SCRIPTS.items()):
            y=28+i*64
            row=F(mod,BORDER,8,y,240,54)
            dot=tk.Label(row,text="●",font=font.Font(family="Helvetica Neue",size=12),
                         bg=BORDER,fg=DIM); dot.place(x=8,y=16)
            tk.Label(row,text=label,font=font.Font(family="Helvetica Neue",size=10),
                     bg=BORDER,fg=TEXT,anchor="w").place(x=30,y=16)
            btn=tk.Button(row,text="Start",font=font.Font(family="Helvetica Neue",size=9),
                          bg=BG,fg=GREEN,bd=0,padx=6,pady=2,cursor="hand2",relief="flat",
                          command=lambda k=key:self._toggle(k))
            btn.place(x=174,y=14)
            self._btns[key]=btn; self._dots[key]=dot

        tk.Button(mod,text="Start All",font=font.Font(family="Helvetica Neue",size=10,weight="bold"),
                  bg=GREEN,fg=BG,bd=0,padx=10,pady=6,cursor="hand2",relief="flat",
                  command=self._start_all).place(x=8,y=440)
        tk.Button(mod,text="Stop All",font=font.Font(family="Helvetica Neue",size=10,weight="bold"),
                  bg=RED,fg=BG,bd=0,padx=10,pady=6,cursor="hand2",relief="flat",
                  command=self._stop_all).place(x=122,y=440)

        # footer
        tk.Frame(self,bg=BORDER,height=1).place(x=0,y=590,width=900)
        ft=F(self,BG,0,593,900,47)
        L(ft,"dir. Gideon Buddenhagen & Noah Zielinski",9,color=DIM).place(x=24,y=12)
        L(ft,"NeuroPawn Knight Board  ·  125Hz  ·  C major",9,color=DIM).place(x=600,y=12)

    def _toggle(self,k):
        if k in self.procs and self.procs[k].poll() is None: self._stop(k)
        else: self._start(k)

    def _start(self,k):
        _,s,a=SCRIPTS[k]
        cmd=f"/opt/homebrew/bin/python3 {os.path.expanduser(s)} {a}".strip()
        try:
            p=subprocess.Popen(cmd,shell=True,stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,preexec_fn=os.setsid)
            self.procs[k]=p
            self._dots[k].config(fg=GREEN); self._btns[k].config(text="Stop",fg=RED)
        except Exception as e: print(e)

    def _stop(self,k):
        if k in self.procs:
            try: os.killpg(os.getpgid(self.procs[k].pid),signal.SIGKILL)
            except: pass
            del self.procs[k]
        self._dots[k].config(fg=DIM); self._btns[k].config(text="Start",fg=GREEN)

    def _start_all(self):
        def seq():
            for k in ["buffer","neuropawn","spectral","music","drums","plot"]:
                self.after(0,lambda k=k:self._start(k)); time.sleep(2)
        threading.Thread(target=seq,daemon=True).start()

    def _stop_all(self):
        for k in list(self.procs): self._stop(k)

    def _vol(self,k,v):
        try:
            if self.reader.r: self.reader.r.set(f"gui.volume.{k}",float(v))
        except: pass

    def _poll(self):
        self.reader.read(); self.n+=1
        self.rdot.config(fg=GREEN if self.reader.connected else RED)
        BW=360
        for band in ("delta","theta","alpha","beta"):
            v,raw=self.reader.ema[band],self.reader.raw_v[band]
            self._bars[band][1].place(width=int(v*BW))
            self._bv[band].config(text=f"{v:.2f}")
            self._br[band].config(text=f"{raw:>12,.0f}")
        a,d,t,b=(self.reader.ema[x] for x in ("alpha","delta","theta","beta"))
        tot=a+d+t+b+0.001
        sc=max(0.0,min(1.0,((a/tot*2.0)+(b/tot*1.5)-(d/tot*1.0))/2.0))
        self._sbar.place(width=int(sc*BW)); self._sv.config(text=f"{sc:.2f}")
        idx=max(0,min(7,int(sc*7)))
        self._cl.config(text=CHORD_NAMES[idx])
        self._sl.config(text=STATE_LABELS[idx])
        self._il.config(text=f"{self.n:04d}")
        self._ml.config(text=self.reader.get_str("gui.last_melody"))
        self._gl.config(text=self.reader.get_str("gui.last_glide"))
        for k in list(self.procs):
            if self.procs[k].poll() is not None:
                self._dots[k].config(fg=RED); self._btns[k].config(text="Start",fg=GREEN)
                del self.procs[k]
        self.after(250,self._poll)

    def on_close(self):
        self._stop_all()
        import time; time.sleep(0.3)
        import subprocess
        subprocess.run("pkill -f brain_music.py; pkill -f drum_synth.py; pkill -f neuropawn_stream.py; pkill -f spectral.py; pkill -f buffer.py; pkill -f plotsignal.py", shell=True)
        self.destroy()
        import sys; sys.exit(0)


if __name__=="__main__":
    app=App()
    app.protocol("WM_DELETE_WINDOW",app.on_close)
    app.mainloop()

#!/usr/bin/env python3
import ctypes as C, ctypes.util, json, math, argparse
from pathlib import Path
f=C.CDLL(ctypes.util.find_library('fluidsynth'))
def bind(n,args,ret):
 x=getattr(f,n);x.argtypes=args;x.restype=ret;return x
P=C.c_void_p;I=C.c_int
newset=bind('new_fluid_settings',[],P);setnum=bind('fluid_settings_setnum',[P,C.c_char_p,C.c_double],I);setint=bind('fluid_settings_setint',[P,C.c_char_p,I],I)
new=bind('new_fluid_synth',[P],P);load=bind('fluid_synth_sfload',[P,C.c_char_p,I],I);unload=bind('fluid_synth_sfunload',[P,I,I],I);select=bind('fluid_synth_program_select',[P,I,I,I,I],I);on=bind('fluid_synth_noteon',[P,I,I,I],I);off=bind('fluid_synth_all_sounds_off',[P,I],I);write=bind('fluid_synth_write_float',[P,I,P,I,I,P,I,I],I);delete=bind('delete_fluid_synth',[P],None);delset=bind('delete_fluid_settings',[P],None)
def synth():
 s=newset();setnum(s,b'synth.sample-rate',22050);setnum(s,b'synth.gain',0.5);setint(s,b'synth.reverb.active',0);setint(s,b'synth.chorus.active',0);setint(s,b'synth.cpu-cores',1);return s,new(s)
def render(s,n,v):
 off(s,0);on(s,0,n,v);N=11025;l=(C.c_float*N)();r=(C.c_float*N)();assert write(s,N,l,0,1,r,0,1)==0;return list(l)+list(r)
parser=argparse.ArgumentParser(description="Compare rendered source and subset SoundFonts")
parser.add_argument("source",type=Path)
parser.add_argument("banks",type=Path)
args=parser.parse_args()
manifest=json.loads((args.banks/'manifest.json').read_text());sa,a=synth();sb,b=synth();aid=load(a,str(args.source).encode(),0);assert aid>=0
report=[]
try:
 for bank in manifest['outputs']:
  bid=load(b,str(args.banks/bank['file']).encode(),0);assert bid>=0
  for p in bank['presets']:
   assert select(a,0,aid,p['bank'],p['program'])==0;assert select(b,0,bid,p['bank'],p['program'])==0
   tests=[]
   for note in ([36,38,42] if p['bank']==128 else [48,60,72]):
    for vel in [32,96]:
     x=render(a,note,vel);y=render(b,note,vel);diff=max(abs(u-v) for u,v in zip(x,y));peak=max(abs(v) for v in y);assert peak>0 and diff==0 and all(math.isfinite(v) for v in y);tests.append({'note':note,'velocity':vel,'peak':peak,'max_abs_diff':diff})
   report.append({'file':bank['file'],**p,'tests':tests});print(p['bank'],p['program'],max(t['max_abs_diff'] for t in tests),flush=True)
  assert unload(b,bid,0)==0
finally:delete(a);delete(b);delset(sa);delset(sb)
(args.banks/'render-report.json').write_text(json.dumps(report,indent=2)+'\n')

#!/usr/bin/env python3
"""Lossless PCM SF2 preset subsetter. Standard library only. No recompression."""
import argparse, hashlib, json, struct
from pathlib import Path

def chunks(b):
    p=0
    while p<len(b):
        if p+8>len(b): raise ValueError('Truncated chunk')
        k,n=struct.unpack_from('<4sI',b,p); v=b[p+8:p+8+n]
        if len(v)!=n: raise ValueError('Truncated chunk payload')
        yield k,v; p+=8+n+(n%2)

def chunk(k,v): return k+struct.pack('<I',len(v))+v+(b'\0' if len(v)%2 else b'')
def records(b,fmt):
    n=struct.calcsize(fmt)
    if len(b)%n: raise ValueError('Bad table size')
    return [list(v) for v in struct.iter_unpack(fmt,b)]
def pack(rs,fmt): return b''.join(struct.pack(fmt,*r) for r in rs)
FMTS={'phdr':'<20sHHHIII','pbag':'<HH','pmod':'<HHhHH','pgen':'<HH','inst':'<20sH','ibag':'<HH','imod':'<HHhHH','igen':'<HH','shdr':'<20sIIIIIBbHH'}
def read(path):
    b=Path(path).read_bytes()
    if b[:4]!=b'RIFF' or b[8:12]!=b'sfbk' or struct.unpack_from('<I',b,4)[0]!=len(b)-8: raise ValueError('Not complete SF2 RIFF')
    lists={v[:4]:list(chunks(v[4:])) for k,v in chunks(b[12:]) if k==b'LIST'}
    pd=dict(lists[b'pdta']); tables={k:records(pd[k.encode()],fmt) for k,fmt in FMTS.items()}
    return b,lists,tables

def subset(src,dst,wanted):
    src,dst=Path(src),Path(dst)
    if src.resolve()==dst.resolve() or dst.exists(): raise ValueError('Refusing overwrite')
    original,lists,t=read(src); sd=dict(lists[b'sdta']); smpl=sd[b'smpl']; sm24=sd.get(b'sm24'); total=len(smpl)//2
    if len(smpl)%2 or (sm24 is not None and len(sm24)<total): raise ValueError('Bad sample data')
    presets=[i for i,r in enumerate(t['phdr'][:-1]) if (r[2],r[1]) in wanted]
    if {(t['phdr'][i][2],t['phdr'][i][1]) for i in presets}!=set(wanted): raise ValueError('Requested preset missing')
    def refs(head,bag,gen,ids,opcode):
        out=set()
        for i in ids:
            for j in range(t[head][i][-1] if head=='inst' else t[head][i][3],t[head][i+1][-1] if head=='inst' else t[head][i+1][3]):
                for op,val in t[gen][t[bag][j][0]:t[bag][j+1][0]]:
                    if op==opcode:out.add(val)
        return out
    instruments=sorted(refs('phdr','pbag','pgen',presets,41)); samples=refs('inst','ibag','igen',instruments,53)
    # Include linked stereo partners, even if only one side is explicitly referenced.
    while True:
        more={t['shdr'][i][8] for i in samples if t['shdr'][i][9]&14}
        if more<=samples:break
        samples|=more
    samples=sorted(samples)
    if any(i>=len(t['shdr'])-1 for i in samples):raise ValueError('Bad sample reference')
    # Preserve generators verbatim; ensure adjusted playback stays inside copied PCM.
    def signed(v): return v if v<32768 else v-65536
    for i in instruments:
        global_gen={}
        for j in range(t['inst'][i][1],t['inst'][i+1][1]):
            g=dict(t['igen'][t['ibag'][j][0]:t['ibag'][j+1][0]])
            if 53 not in g:
                global_gen.update(g);continue
            effective={**global_gen,**g};h=t['shdr'][g[53]]
            def adj(fine,coarse):return signed(effective.get(fine,0))+32768*signed(effective.get(coarse,0))
            start=h[1]+adj(0,4);end=h[2]+adj(1,12)
            ls=h[3]+adj(2,45);le=h[4]+adj(3,50)
            if not h[1]<=start<end<=h[2] or not h[1]<=ls<=le<=h[2]:
                raise ValueError('Address generator accesses outside copied sample')
    imap={v:i for i,v in enumerate(instruments)};smap={v:i for i,v in enumerate(samples)}
    out={};pcm=bytearray();pcm24=bytearray();headers=[];sample_checks=[]
    for i in samples:
        r=t['shdr'][i].copy(); start,end,ls,le=r[1:5]
        if r[9]&0x8010: raise ValueError('ROM/compressed SF3 samples unsupported')
        if not 0<=start<=ls<=le<=end<=total:raise ValueError('Invalid sample loop')
        new=len(pcm)//2; segment=smpl[start*2:end*2];pcm+=segment+b'\0'*92
        if sm24 is not None:pcm24+=sm24[start:end]+b'\0'*46
        r[1:5]=[new,new+end-start,new+ls-start,new+le-start]
        if r[9]&14:r[8]=smap[r[8]]
        else:r[8]=0
        headers.append(r);sample_checks.append({'source_index':i,'name':r[0].rstrip(b'\0').decode('latin1'),'frames':end-start,'pcm_sha256':hashlib.sha256(segment).hexdigest()})
    headers.append([b'EOS',len(pcm)//2,len(pcm)//2,len(pcm)//2,len(pcm)//2,0,0,0,0,1]);out['shdr']=headers
    def rebuild(head,bag,mod,gen,ids,op,remap):
        hs=[];bs=[];ms=[];gs=[];idx=1 if head=='inst' else 3
        for i in ids:
            r=t[head][i].copy();r[idx]=len(bs);hs.append(r)
            for j in range(t[head][i][idx],t[head][i+1][idx]):
                bs.append([len(gs),len(ms)]);ms.extend(t[mod][t[bag][j][1]:t[bag][j+1][1]])
                for g in t[gen][t[bag][j][0]:t[bag][j+1][0]]:gs.append([g[0],remap[g[1]] if g[0]==op else g[1]])
        term=t[head][-1].copy();term[idx]=len(bs);hs.append(term);bs.append([len(gs),len(ms)]);gs.append([0,0]);ms.append([0]*5)
        out.update({head:hs,bag:bs,mod:ms,gen:gs})
    rebuild('phdr','pbag','pmod','pgen',presets,41,imap);rebuild('inst','ibag','imod','igen',instruments,53,smap)
    data=chunk(b'LIST',b'INFO'+b''.join(chunk(k,v) for k,v in lists[b'INFO']))
    data+=chunk(b'LIST',b'sdta'+chunk(b'smpl',pcm)+(chunk(b'sm24',pcm24) if sm24 is not None else b''))
    data+=chunk(b'LIST',b'pdta'+b''.join(chunk(k.encode(),pack(out[k],fmt)) for k,fmt in FMTS.items()))
    dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(chunk(b'RIFF',b'sfbk'+data))
    _,again,u=read(dst)
    assert again[b'INFO']==lists[b'INFO']
    for j,i in enumerate(samples):
        h=u['shdr'][j];assert bytes(pcm[h[1]*2:h[2]*2])==smpl[t['shdr'][i][1]*2:t['shdr'][i][2]*2]
    return {'file':dst.name,'bytes':dst.stat().st_size,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'presets':[{'bank':t['phdr'][i][2],'program':t['phdr'][i][1],'name':t['phdr'][i][0].rstrip(b'\0').decode('latin1')} for i in presets],'instruments':len(instruments),'samples':sample_checks}
GROUPS={'piano':[0],'keys':[2,8,16],'guitars':[24,26,29,32,33],'strings':[40,41,42,43,46,47,48,50,52],'winds':[56,57,58,60,61,64,66,67,68,69,70,71,72],'synth':[80,88,100,101]}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source');ap.add_argument('--out',required=True);ap.add_argument('--group',choices=[*GROUPS,'drums','all'],default='all');args=ap.parse_args()
    groups={k:[(0,p) for p in v] for k,v in GROUPS.items()};groups['drums']=[(128,0)]
    manifest={'source':Path(args.source).name,'source_sha256':hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),'excluded_gm_program':55,'outputs':[]}
    for name,ids in groups.items():
        if args.group in ('all',name):manifest['outputs'].append(subset(args.source,Path(args.out)/f'MuseScore_General_{name}.sf2',ids))
    Path(args.out,'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps([{k:x[k] for k in ['file','bytes','instruments']} for x in manifest['outputs']],indent=2))

import json,glob,collections,re,sys,csv
base=sys.argv[1] if len(sys.argv)>1 else '.'
def norm(s):
    s=(s or '').lower().replace('&','and').replace("’","'")
    s=re.sub(r'^(ex|sv\d*|swsh\d*|sm|xy|bw|dp|hgss|me\d*)[: -]+','',s)
    s=re.sub(r'(pokemon|tcg|series|promos?|black star|cards?|collection)','',s)
    return re.sub(r'[^a-z0-9]','',s)
def num(x):
    x=str(x or '').split('/')[0].strip().lower()
    m=re.match(r'^([a-z]*)0*(\d+)([a-z]*)$',x)
    return (m.group(1)+m.group(2)+m.group(3)) if m else x
rows=[json.loads(l) for l in open(base+'/tcgdex/cards.jsonl',encoding='utf-8')]
tp={};idx=collections.defaultdict(list)
for r in rows:
    for v in r.get('variants_detailed',[]) or []:
        t=(v.get('thirdParty') or {}).get('tcgplayer')
        if t: tp[str(t)]=r
    idx[(norm(r['set'].get('name')),num(r.get('localId')))].append(r)
setnames={}
for f in glob.glob(base+'/raw/*.json'):
    for c in json.load(open(f,encoding='utf-8')).get('data',[]):
        setnames[c['set']]=c.get('set_name')
out=[];st=collections.Counter();un=collections.Counter()
for f in glob.glob(base+'/raw/*.json'):
    for c in json.load(open(f,encoding='utf-8')).get('data',[]):
        if str(c.get('number'))=='N/A': st['sealed']+=1;continue
        r=tp.get(str(c.get('tcgplayerId')));how='id'
        if not r:
            cand=idx.get((norm(c.get('set_name')),num(c.get('number'))),[])
            if len(cand)==1: r=cand[0];how='set+no'
            elif len(cand)>1:
                nm=[x for x in cand if norm(x.get('name'))==norm(c.get('name'))]
                if len(nm)==1: r=nm[0];how='set+no+name'
        if r: st[how]+=1; out.append((c['id'],c['tcgplayerId'],r['id'],how))
        else: st['yok']+=1; un[c.get('set_name')]+=1
print(dict(st));print(un.most_common(15))
w=csv.writer(open(base+'/../outputs/justtcg_tcgdex_map.csv','w',newline='',encoding='utf-8'))
w.writerow(['justtcg_id','tcgplayer_id','tcgdex_id','how']);w.writerows(out)

import base64
import sys
from pathlib import Path

import cv2
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.pipeline import Pipeline
from src.common import config

UPLOAD = config.DEMO_DIR / 'upload.jpg'

CSS = """
.stApp {background:linear-gradient(180deg,#22365E 0%,#1B2A4A 100%) !important;}
[data-testid="stAppViewContainer"] {background:transparent !important;}
header[data-testid="stHeader"] {background:#D7261E;border-bottom:3px solid #FFCB05;}
[data-testid="stSidebar"] {background:#16233F;border-right:4px solid #FFCB05;}
[data-testid="stSidebar"] h3, .main h2, .main h3 {color:#FFCB05 !important;}
[data-testid="stFileUploader"] section {background:#243A63;border:2px dashed #FFCB05;border-radius:14px;}
[data-testid="stContainer"] {border-radius:14px;}
[data-testid="stVerticalBlockBorderWrapper"] {border:2px solid #FFCB05 !important;border-radius:16px !important;background:#243A63;}
[data-testid="stStatus"] {border:2px solid #FFCB05 !important;border-radius:12px;}
button[kind="primary"] {background:#D7261E !important;border:2px solid #FFCB05 !important;color:#ffffff !important;font-weight:700;}
button[kind="secondary"] {border:2px solid #FFCB05 !important;color:#F7F3E8 !important;background:transparent !important;}
.stSlider [data-baseweb="slider"] div[role="slider"] {background:#D7261E !important;border:2px solid #FFCB05 !important;}
.hero {background:#D7261E;border:3px solid #FFCB05;border-radius:16px;padding:16px 22px;margin-bottom:14px;display:flex;align-items:center;gap:18px;}
.hero img {width:72px;height:72px;border-radius:16px;border:2px solid #FFCB05;}
.hero h2 {margin:0;padding:0;color:#ffffff !important;font-weight:800;font-size:34px;line-height:1.1;}
.hero h2 span {color:#ffffff !important;}
.hero h2 .b2, .side-brand .b2 {color:#FFCB05 !important;}
.hero p {margin:4px 0 0 0;color:#FFF3C4;}
.side-brand {display:flex;align-items:center;gap:10px;margin-bottom:6px;}
.side-brand img {width:40px;height:40px;border-radius:10px;}
.side-brand b {font-size:22px;color:#F7F3E8;}
.match {font-size:12px;color:#FFCB05;font-weight:700;min-height:18px;}
.badge {display:inline-block;padding:3px 12px;border-radius:999px;font-size:12px;font-weight:700;}
.badge-kart {background:#2e7d32;color:#ffffff;}
.badge-model {background:#FFCB05;color:#1B2A4A;}
.price-card {background:#F7F3E8;color:#1B2A4A;border:4px solid #FFCB05;border-radius:18px;padding:18px 22px;box-shadow:0 4px 14px rgba(0,0,0,0.15);}
.price-big {font-size:44px;font-weight:800;line-height:1.1;margin-top:8px;}
.range-bar {position:relative;height:12px;border-radius:6px;background:linear-gradient(90deg,#90caf9,#FFCB05,#D7261E);margin:18px 4px 6px 4px;}
.range-marker {position:absolute;top:-6px;width:6px;height:24px;background:#1B2A4A;border-radius:3px;}
.range-labels {display:flex;justify-content:space-between;font-size:13px;color:#1B2A4A;}
.tip {font-size:14px;}
.gg-title {font-size:13px;color:#FFCB05;font-weight:700;margin:4px 0 8px 0;}
.gg-row {display:flex;gap:8px;}
.gg {flex:1;text-align:center;opacity:0.5;}
.gg.active {opacity:1;}
.gg-score {background:#243A63;border:2px solid #F7F3E8;border-radius:10px;padding:6px 0;font-size:20px;font-weight:800;color:#F7F3E8;}
.gg.active .gg-score {background:#D7261E;border-color:#FFCB05;}
.gg-arrow {color:#FFCB05;font-size:11px;line-height:1.5;}
.gg-cond {background:#F7F3E8;color:#1B2A4A;border-radius:10px;padding:5px 2px;font-size:11px;display:flex;flex-direction:column;}
.gg.active .gg-cond {border:2px solid #FFCB05;}
.gg-cond b {font-size:13px;}
"""


def set_choice(i):
    st.session_state.choice = i


def price_html(f, grade):
    if f['kaynak'] == 'kart':
        badge_cls, badge_txt = 'badge-kart', 'Kartın gerçek referans fiyatı'
    else:
        badge_cls, badge_txt = 'badge-model', 'Model tahmini'
    est, low, high = f['estimate'], f['range_low'], f['range_high']
    pos = (est - low) / (high - low) * 100 if high > low else 50
    pos = min(max(pos, 0), 100)
    return """<div class="price-card">
<span class="badge {cls}">{txt}</span>
<div class="tip" style="margin-top:10px;">Derece {grade} · Kondisyon {cond}</div>
<div class="price-big">{est:.2f} EUR</div>
<div class="range-bar"><div class="range-marker" style="left:calc({pos:.1f}% - 3px)"></div></div>
<div class="range-labels"><span>{low:.2f} EUR</span><span>{high:.2f} EUR</span></div>
<div class="tip" style="margin-top:10px;">Referans fiyat: {ref:.2f} EUR</div>
</div>""".format(
        cls=badge_cls, txt=badge_txt, grade=grade, cond=f['condition'],
        est=est, pos=pos, low=low, high=high, ref=f['reference_price'],
    )


GRADE_GROUPS = [
    ('10', 10, 10, 'NM', 'Near Mint'),
    ('9–8', 8, 9, 'LP', 'Lightly Played'),
    ('7–6', 6, 7, 'MP', 'Moderately Played'),
    ('5–4', 4, 5, 'HP', 'Heavily Played'),
    ('3–1', 1, 3, 'DMG', 'Damaged'),
]


def grade_guide_html(grade):
    cells = []
    for label, lo, hi, short, long_name in GRADE_GROUPS:
        active = ' active' if lo <= grade <= hi else ''
        cells.append('<div class="gg{a}"><div class="gg-score">{l}</div><div class="gg-arrow">▼</div><div class="gg-cond"><b>{s}</b><span>{n}</span></div></div>'.format(a=active, l=label, s=short, n=long_name))
    return '<div class="gg-title">PSA tarzı derece → fiyat tablosundaki kondisyon</div><div class="gg-row">' + ''.join(cells) + '</div>'


def logo_tag():
    if not config.LOGO.exists():
        return ''
    data = base64.b64encode(config.LOGO.read_bytes()).decode()
    return '<img src="data:image/png;base64,%s" alt="%s">' % (data, config.BRAND)


def brand_html():
    return '%s<span class="b2">%s</span>' % (config.BRAND[:4], config.BRAND[4:])


def match_html(c):
    tags = []
    if c.get('ad_eslesti'):
        tags.append('✓ Ad')
    if c.get('numara_eslesti'):
        tags.append('✓ Numara')
    return '<div class="match">%s</div>' % ' · '.join(tags)


@st.cache_resource(show_spinner=False)
def get_pipeline():
    return Pipeline()


st.set_page_config(
    page_title='%s · Kart Fiyat Tahmini' % config.BRAND,
    page_icon=str(config.LOGO) if config.LOGO.exists() else '🃏',
    layout='wide',
)
st.markdown('<style>' + CSS + '</style>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="side-brand">%s<b>%s</b></div>' % (logo_tag(), brand_html()), unsafe_allow_html=True)
    st.markdown('### Çekim önerileri')
    st.markdown(
        '- İyi ve dağınık olmayan ışık kullanın\n'
        '- Parlamayan bir yüzey seçin\n'
        '- Sade, tek renkli bir arka plan kullanın\n'
        '- Kartın dört kenarını karede tutun\n'
        '- Kartı düz tutun, eğik çekmeyin\n'
        '- Kartın altındaki numara okunaklı olsun'
    )
    st.markdown('### Nasıl çalışır')
    st.markdown(
        '1. Fotoğrafı yükleyin\n'
        '2. Aday kartlardan birini seçin\n'
        '3. Dereceyi belirleyin\n'
        '4. Fiyat tahminini görün'
    )
    st.caption('Fiyatlar EUR cinsindendir ve TCGdex / Cardmarket verisine dayanır.')

st.markdown(
    '<div class="hero">%s<div><h2>%s</h2>'
    '<p>%s Pokémon kartının fotoğrafını yükleyin, kartınızı seçin, dereceyi belirleyin.</p></div></div>'
    % (logo_tag(), brand_html(), config.TAGLINE),
    unsafe_allow_html=True,
)

file = st.file_uploader('Kart fotoğrafını yükleyin', type=['jpg', 'jpeg', 'png'])
if file is None:
    st.stop()

UPLOAD.parent.mkdir(parents=True, exist_ok=True)
UPLOAD.write_bytes(file.getvalue())

with st.spinner('Modeller yükleniyor...'):
    p = get_pipeline()

with st.status('Kart analiz ediliyor...', expanded=True) as status:
    status.write('Köşeler bulunuyor ve kart düzeltiliyor')
    res = p.analyze(UPLOAD)
    if 'hata' in res:
        status.update(label='Analiz tamamlanamadı', state='error')
    else:
        status.write('Benzer kartlar arandı, kart adı ve numarası okundu')
        status.update(label='Analiz tamam', state='complete', expanded=False)

if 'hata' in res:
    st.error(res['hata'])
    st.stop()

cands = res['adaylar']

if st.session_state.get('upload_name') != file.name:
    st.session_state.upload_name = file.name
    st.session_state.choice = 0
if st.session_state.get('choice', 0) >= len(cands):
    st.session_state.choice = 0
choice = st.session_state.choice

left, right = st.columns([1, 1])
with left:
    st.markdown('**Düzeltilmiş kart**')
    st.image(cv2.cvtColor(res['kirpilmis'], cv2.COLOR_BGR2RGB), width=260)
with right:
    st.markdown('**Derece**')
    grade = st.slider('Kart derecesi (1-10)', min_value=1, max_value=10, value=8, label_visibility='collapsed')
    st.caption('Derece, kartın kondisyonunu tahmin etmek için kullanılır.')
    st.markdown(grade_guide_html(grade), unsafe_allow_html=True)

st.markdown('### Aday kartlar')
cols = st.columns(len(cands))
for i, (col, c) in enumerate(zip(cols, cands)):
    img = p.meta.get(c['id'], {}).get('raw', {}).get('image')
    selected = choice == i
    with col:
        with st.container(border=True):
            if img:
                st.image(img + '/low.webp', width=150)
            st.markdown('**%s**' % c['name'])
            st.caption('%s · %s' % (c['set'], c['number']))
            st.markdown(match_html(c), unsafe_allow_html=True)
            st.progress(min(max(c['benzerlik'], 0.0), 1.0), text='Benzerlik %.2f' % c['benzerlik'])
            st.button(
                '✔ Seçili' if selected else 'Bu kartı seç',
                key='btn_%d' % i,
                on_click=set_choice,
                args=(i,),
                type='primary' if selected else 'secondary',
            )

st.markdown('### Fiyat')
fiyat = p.price_for(cands[choice]['id'], grade)
if fiyat is None:
    st.info('Bu kart için fiyat hesaplanamadı.')
else:
    if fiyat.get('uyari'):
        st.warning(fiyat['uyari'])
    st.markdown(price_html(fiyat, grade), unsafe_allow_html=True)
    if fiyat['kaynak'] == 'model' and fiyat.get('range_level'):
        st.caption('Aralık, benzer kartlarda gerçek fiyatın yaklaşık %%%d ihtimalle düştüğü bölgedir; kesin fiyat değildir.' % round(fiyat['range_level'] * 100))
    else:
        st.caption('Aralık, aynı kondisyondaki kartlarda görülen fiyat yayılımını gösterir; kesin fiyat değildir.')

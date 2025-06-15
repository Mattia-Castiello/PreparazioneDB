import os
import json
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from PIL import Image

# ====== CONFIGURAZIONE ======
dir_json = '/Users/mattiacastiello/Desktop/tesi/code/PreparazioneDB/Filejson'
xlsx_path = '/Users/mattiacastiello/Desktop/tesi/code/PreparazioneDB/ValutazioneAnnotazioni copia.xlsx'
images_dir = '/Users/mattiacastiello/Desktop/tesi/code/PreparazioneDB/Images'  # cartella con le immagini
output_dir = dir_json
output_filename = 'merged_annotations.json'

# ====== UTILITIES ======
def debug(msg):
    print(f'[DEBUG] {msg}')

# ====== VERIFICA INPUT ======
if not os.path.isdir(dir_json):
    raise FileNotFoundError(f'Directory non trovata: {dir_json}')
all_files = os.listdir(dir_json)
json_files = [f for f in all_files if f.lower().endswith('.json') and f != output_filename]
if not json_files:
    raise FileNotFoundError(f'Nessun JSON trovato in {dir_json} (escluso {output_filename})')
debug(f'Trovati {len(json_files)} file JSON di input')

# ====== LETTURA EXCEL ======
df = pd.read_excel(xlsx_path, dtype=str)
debug(f'Colonne Excel: {list(df.columns)}')
required = {'id_annotatore', 'punteggio', 'id_annotatore_equivalente'}
if not required.issubset(df.columns):
    raise KeyError(f'Colonne mancanti in Excel: {required}')

# Costruzione mapping annotatori equivalenti e punteggi
temp_map = {}
score_mapping = {}
for _, row in df.iterrows():
    base = str(row['id_annotatore']).strip()
    eqs = str(row['id_annotatore_equivalente']).strip()
    eqs_clean = eqs.replace(';', ',')
    temp_map[base] = [] if eqs_clean.lower() in ('nan', '') else [e.strip() for e in eqs_clean.split(',') if e.strip()]
    try:
        score_mapping[base] = float(row['punteggio'])
    except:
        score_mapping[base] = 0.0
mapping = temp_map
debug(f'Mappatura equivalenti e punteggi caricati per {len(mapping)} annotatori')

# ====== CARICAMENTO JSON ======
records = []
for fname in json_files:
    path = os.path.join(dir_json, fname)
    data = json.load(open(path, 'r'))
    annot_id = data.get('id_annotatore') or data.get('annotations', [{}])[0].get('id_annotatore')
    if annot_id is None:
        annot_id = os.path.splitext(fname)[0]
        debug(f'Nessun id_annotatore in {fname}; uso nome file: {annot_id}')
    annot_id = str(annot_id).strip()
    if annot_id.isdigit():
        cand = f'id{int(annot_id)}'
        if cand in score_mapping:
            annot_id = cand
            debug(f'Normalizzato {annot_id}')
    score = score_mapping.get(annot_id, 0.0)
    if annot_id not in score_mapping:
        debug(f'Annotatore {annot_id} non trovato in Excel; punteggio=0')
    records.append({'annot_id': annot_id, 'score': score, 'data': data})
debug(f'Caricati {len(records)} record JSON con punteggi associati')

# ====== FASE 1: UNIONE PER GRUPPO ======
groups = []
processed = set()
for rec in records:
    aid = rec['annot_id']
    if aid in processed:
        continue
    group_ids = [aid] + mapping.get(aid, [])
    group_recs = [r for r in records if r['annot_id'] in group_ids]
    processed.update(r['annot_id'] for r in group_recs)
    max_score = max(r['score'] for r in group_recs)
    top_recs = [r for r in group_recs if r['score'] == max_score]
    debug(f'Gruppo {group_ids}: {len(top_recs)} top recs (score={max_score})')
    imgs = [img for r in top_recs for img in r['data'].get('images', [])]
    cats = [cat for r in top_recs for cat in r['data'].get('categories', [])]
    seen_fn, uniq_imgs = set(), []
    for img in imgs:
        if img['file_name'] not in seen_fn:
            seen_fn.add(img['file_name']); uniq_imgs.append(img)
    seen_cat, uniq_cats = set(), []
    for cat in cats:
        if cat['name'] not in seen_cat:
            seen_cat.add(cat['name']); uniq_cats.append(cat)
    if len(top_recs) == 1:
        anns = top_recs[0]['data'].get('annotations', [])
    else:
        temp = [a for r in top_recs for a in r['data'].get('annotations', [])]
        used, merged = set(), []
        for i, a in enumerate(temp):
            if a['id'] in used:
                continue
            paired = False
            for b in temp[i+1:]:
                if b['id'] in used or a['image_id'] != b['image_id']:
                    continue
                sa = a.get('segmentation', [])
                sb = b.get('segmentation', [])
                if not sa or not sb or not isinstance(sa, list) or not isinstance(sb, list):
                    continue
                coords_a = sa[0]
                coords_b = sb[0]
                if len(coords_a) < 6 or len(coords_b) < 6:
                    continue
                try:
                    pa = Polygon([(coords_a[k], coords_a[k+1]) for k in range(0, len(coords_a), 2)])
                    pb = Polygon([(coords_b[k], coords_b[k+1]) for k in range(0, len(coords_b), 2)])
                except ValueError:
                    continue
                if not (pa.is_valid and pb.is_valid):
                    continue
                iou = pa.intersection(pb).area / pa.union(pb).area
                if iou > 0.7:
                    inter = pa.intersection(pb)
                    ext = list(inter.exterior.coords)
                    merged.append({
                        'id': max(a['id'], b['id']) + 1,
                        'image_id': a['image_id'],
                        'category_id': a['category_id'],
                        'segmentation': [sum(([pt[0], pt[1]] for pt in ext), [])],
                        'bbox': list(inter.bounds),
                        'area': inter.area,
                        'ignore': a.get('ignore', 0),
                        'iscrowd': a.get('iscrowd', 0),
                        'id_annotatore': a.get('id_annotatore')
                    })
                    used.update({a['id'], b['id']})
                    paired = True
                    break
            if not paired and a['id'] not in used:
                merged.append(a)
                used.add(a['id'])
        anns = merged
    groups.append({'images': uniq_imgs, 'categories': uniq_cats, 'annotations': anns})
debug(f'Costruiti {len(groups)} gruppi combinati')

# ====== FASE 2: COSTRUZIONE OUTPUT GLOBALE ======
final_images, image_map = [], {}
final_categories, category_map = [], {}
final_annotations = []
next_img_id = next_cat_id = next_ann_id = 0

for grp in groups:
    local_cat = {c['id']: c['name'] for c in grp['categories']}
    for name in local_cat.values():
        if name not in category_map:
            category_map[name] = next_cat_id
            final_categories.append({'id': next_cat_id, 'name': name})
            next_cat_id += 1
    local_img = {img['id']: img['file_name'] for img in grp['images']}
    for img in grp['images']:
        fn = img['file_name']
        if fn not in image_map:
            image_map[fn] = next_img_id
            final_images.append({'width': img['width'], 'height': img['height'], 'id': next_img_id, 'file_name': fn})
            next_img_id += 1
    for a in grp['annotations']:
        fn = local_img.get(a['image_id'])
        if fn is None:
            continue
        final_annotations.append({
            'id': next_ann_id,
            'image_id': image_map[fn],
            'category_id': category_map[local_cat[a['category_id']]],
            'segmentation': a.get('segmentation', []),
            'bbox': a.get('bbox', []),
            'area': a.get('area', 0),
            'ignore': a.get('ignore', 0),
            'iscrowd': a.get('iscrowd', 0),
            'id_annotatore': a.get('id_annotatore')
        })
        next_ann_id += 1

debug(f'Final counts: images={len(final_images)}, categories={len(final_categories)}, annotations={len(final_annotations)}')

# ====== SALVATAGGIO ======
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, output_filename), 'w') as f:
    json.dump({'images': final_images, 'categories': final_categories, 'annotations': final_annotations}, f, indent=2)
debug('Merged annotations salvate correttamente')

# ====== VISUALIZZAZIONE SEGMENTAZIONI ======
# Carica il file JSON fuso
data = json.load(open(os.path.join(output_dir, output_filename), 'r'))
# Mappa id immagine -> nome file completo
id2file = {img['id']: img['file_name'] for img in data['images']}
# Raggruppa annotazioni per immagine
anns_by_img = {}
for ann in data['annotations']:
    anns_by_img.setdefault(ann['image_id'], []).append(ann)

# Per ogni immagine, disegna tutte le segmentazioni sovrapposte
for img_id, anns in anns_by_img.items():
    raw_fn = id2file[img_id]
    base_fn = os.path.basename(raw_fn)
    # estrai il nome reale (parte dopo il primo trattino)
    real_fn = base_fn.split('-', 1)[-1] if '-' in base_fn else base_fn
    img_path = os.path.join(images_dir, real_fn)
    if not os.path.isfile(img_path):
        debug(f"Immagine non trovata: {img_path}")
        continue
    img = Image.open(img_path).convert("RGB")
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    for ann in anns:
        for seg in ann.get('segmentation', []):
            coords = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
            xs, ys = zip(*coords)
            plt.plot(xs + (xs[0],), ys + (ys[0],), linewidth=2)
    plt.axis('off')
    plt.title(f"{real_fn} - {len(anns)} segmentazioni")
    plt.show()
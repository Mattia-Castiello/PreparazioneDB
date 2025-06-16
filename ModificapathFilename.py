import json
import os

# Percorso alla cartella dove si trovano i JSON da correggere
json_dir = '/Users/mattiacastiello/Desktop/tesi/code/PreparazioneDB/Filejson'
# Lista dei file JSON da aggiornare
json_files = ['merged_annotations.json']
# Nuovo prefisso per le immagini
new_prefix = '/Users/mattiacastiello/Desktop/tesi/code/PreparazioneDB/Images'

for jf in json_files:
    path = os.path.join(json_dir, jf)
    with open(path, 'r') as f:
        data = json.load(f)
    for img in data.get('images', []):
        orig = img.get('file_name', '')
        # normalizziamo tutti i separatori a '/'
        orig_norm = orig.replace('\\', '/')
        # prendiamo solo il nome del file
        base = os.path.basename(orig_norm)
        # se il nome contiene un uuid-separatore, usiamo la parte dopo il primo '-'
        if '-' in base:
            new_name = base.split('-', 1)[1]
        else:
            new_name = base
        img['file_name'] = os.path.join(new_prefix, new_name)
    # riscriviamo il JSON aggiornato
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Aggiornato {jf}')
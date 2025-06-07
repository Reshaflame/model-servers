import json, csv, gzip, os

META_DIR = "data/meta"; os.makedirs(META_DIR, exist_ok=True)

def save_category_maps(auth_types, logon_types, orientations):
    with open(f"{META_DIR}/auth_type_map.json","w") as f:
        json.dump({v:i for i,v in enumerate(sorted(auth_types))}, f)
    with open(f"{META_DIR}/logon_type_map.json","w") as f:
        json.dump({v:i for i,v in enumerate(sorted(logon_types))}, f)
    with open(f"{META_DIR}/auth_orientation_map.json","w") as f:
        json.dump({v:i for i,v in enumerate(sorted(orientations))}, f)

def save_feature_list(cols):
    with open(f"{META_DIR}/expected_features.json","w") as f:
        json.dump(cols, f, indent=2)

def dump_freq_dict(path, d):
    with gzip.open(path,"wt",newline='') as gz:
        w = csv.writer(gz)
        w.writerow(["key","count"])
        for k,c in d.items(): w.writerow([k,c])

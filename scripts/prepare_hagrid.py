"""Prépare le dataset YOLO de l'approche B : HaGRID (scènes réelles) + vos propres images.

Utilisation :
    python scripts/prepare_hagrid.py --hagrid-ann <dossier annotations> --hagrid-img <dossier images>
    python scripts/prepare_hagrid.py --own-only          (uniquement vos images de collect_data.py)

HaGRID (https://github.com/hukenovs/hagrid) : un fichier JSON par geste,
    {image_id: {"bboxes": [[x, y, w, h], ...] (normalisées, coin haut-gauche), "labels": [...], "user_id": ...}}
Les versions complète, « sample 30k » (Kaggle) et HaGRIDv2 ont cette structure : le script cherche
les JSON et les images récursivement, quel que soit le découpage des dossiers.

Classes YOLO = config.LABELS (none, palm, thumbs_up, thumbs_down, two_fingers, index_left, fist).
« index_left » n'existe pas dans HaGRID : il vient de vos images (data/images, collect_data.py).
Sortie : data/yolo/{images,labels}/{train,val}/ + data/yolo/data.yaml
"""
import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maestro import config  # noqa: E402

CLASS_ID = {name: i for i, name in enumerate(config.LABELS)}
IMG_EXT = {".jpg", ".jpeg", ".png"}


def map_hagrid(label: str) -> str | None:
    if label in config.HAGRID_MAP:
        return config.HAGRID_MAP[label]
    if label in config.HAGRID_NONE_CLASSES:
        return config.NONE
    return None   # geste ambigu (peace_inverted, three2...) : image ignorée


def yolo_line(cls: str, x0, y0, x1, y1) -> str:
    x0, y0, x1, y1 = (min(max(v, 0.0), 1.0) for v in (x0, y0, x1, y1))
    return f"{CLASS_ID[cls]} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"


def copy_image(src: Path, dst: Path, max_side: int):
    """Copie l'image en la réduisant (économise le disque ; les boîtes normalisées ne changent pas)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(src))
    if img is None:
        return False
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        k = max_side / max(h, w)
        img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        shutil.copy2(src, dst)
    return True


def hagrid_samples(ann_dir: Path, img_dir: Path, per_class: int, rng: random.Random):
    """-> liste de (chemin image, [lignes YOLO], groupe pour le découpage, classe source)."""
    print("Indexation des images HaGRID...")
    images = {p.stem: p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT}
    print(f"  {len(images)} images trouvées")
    by_class = defaultdict(list)
    for jf in sorted(ann_dir.rglob("*.json")):
        source = jf.stem
        if map_hagrid(source) is None:
            continue
        for image_id, ann in json.loads(jf.read_text(encoding="utf-8")).items():
            path = images.get(image_id)
            if path is None:
                continue
            mapped = [map_hagrid(lab) for lab in ann.get("labels", [])]
            if not mapped or None in mapped:
                continue
            lines = [yolo_line(m, x, y, x + w, y + h) for m, (x, y, w, h) in zip(mapped, ann["bboxes"])]
            by_class[source].append((path, lines, str(ann.get("user_id", image_id)), source))
    out = []
    for source, items in sorted(by_class.items()):
        rng.shuffle(items)
        out += items[:per_class]
        print(f"  {source:<12} -> {map_hagrid(source):<12} {min(len(items), per_class)} images")
    return out


def own_samples(csv_path: Path):
    """Vos images (collect_data.py) : une main annotée par image, groupe = session d'enregistrement."""
    out = []
    if not csv_path.exists():
        return out
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("image"):
                continue
            path = config.DATA_DIR / row["image"]
            if path.exists():
                box = [float(row[k]) for k in ("x_min", "y_min", "x_max", "y_max")]
                out.append((path, [yolo_line(row["label"], *box)], path.stem.split("_")[0], "maestro"))
    print(f"  vos images : {len(out)}")
    return out


def split_by_group(samples, val_frac: float, rng: random.Random):
    """Découpage par personne / session : aucune personne n'est à la fois en train et en val."""
    groups = sorted({s[2] for s in samples})
    rng.shuffle(groups)
    if len(groups) < 3:   # trop peu de groupes : découpage aléatoire
        idx = list(range(len(samples)))
        rng.shuffle(idx)
        val = set(idx[:round(len(idx) * val_frac)])
        return [s for i, s in enumerate(samples) if i not in val], [s for i, s in enumerate(samples) if i in val]
    val_groups = set(groups[:max(1, round(len(groups) * val_frac))])
    return [s for s in samples if s[2] not in val_groups], [s for s in samples if s[2] in val_groups]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hagrid-ann", type=Path, help="dossier des annotations JSON HaGRID")
    parser.add_argument("--hagrid-img", type=Path, help="dossier des images HaGRID")
    parser.add_argument("--own-only", action="store_true", help="n'utiliser que vos images")
    parser.add_argument("--per-class", type=int, default=800, help="images HaGRID gardées par classe source")
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--out", type=Path, default=config.YOLO_DATASET_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    samples = []
    if not args.own_only:
        if not (args.hagrid_ann and args.hagrid_img):
            sys.exit("Indiquez --hagrid-ann et --hagrid-img (ou --own-only).")
        samples += hagrid_samples(args.hagrid_ann, args.hagrid_img, args.per_class, rng)
    own = own_samples(config.DATASET_CSV)
    if not own:
        print("  (aucune image perso : « index_left » n'aura pas d'exemples -> lancez collect_data.py)")
    samples += own
    if not samples:
        sys.exit("Aucune image à préparer.")

    if args.out.exists():
        shutil.rmtree(args.out)
    counts = {"train": Counter(), "val": Counter()}
    for split, items in zip(("train", "val"), split_by_group(samples, args.val, rng)):
        for i, (src, lines, _, source) in enumerate(items):
            name = f"{source}_{i:06d}"
            if not copy_image(src, args.out / "images" / split / f"{name}.jpg", args.max_side):
                continue
            lbl = args.out / "labels" / split / f"{name}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("\n".join(lines) + "\n")
            counts[split].update(config.LABELS[int(ln.split()[0])] for ln in lines)

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(config.LABELS))
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    for split, c in counts.items():
        print(f"{split:<5} boîtes par classe : {dict(c)}")
    print(f"Dataset YOLO prêt : {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()

"""Plot the repartition of `unique_label` (infrared images by default) with a 2x2 mosaic
of 4 example images above each bar, and sort all the images into DATA/global/unique_labels/<label>/ folders.
With --multi, also plot the repartition of `multi_labels` (an image counts once for each of its
labels) with 4 example images per label.

Usage:
    python Experiment_scripts/unique_labels.py
    python Experiment_scripts/unique_labels.py --example urban=000003_infrared.png,000100_infrared.png --example people=000231_infrared.png
    python Experiment_scripts/unique_labels.py --modality all --no-sort --save-dir Figures
    python Experiment_scripts/unique_labels.py --multi --multi-example car=000001_infrared.png,000050_infrared.png
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1] / "DATA_OCCULO_2026" / "2026" / "global"
METADATA = ROOT / "metadata" / "metadata.json"
IMAGES = ROOT / "images"
OUT_DIR = ROOT / "unique_labels"


def image_path(entry):
    return IMAGES / entry["modality"] / entry["name"]


def multi_labels(entry):
    return [label.strip() for label in entry["multi_labels"].split(",") if label.strip()]


def parse_examples(pairs):
    """Turn ['label=image1,image2', ...] into {label: [image1, image2]}."""
    examples = {}
    for pair in pairs or []:
        label, sep, names = pair.partition("=")
        if not sep:
            raise ValueError(f"Examples must be 'label=image_name[,image_name...]', got '{pair}'")
        examples[label.strip()] = [name.strip() for name in names.split(",") if name.strip()]
    return examples


def pick_examples(metadata, labels, examples, get_labels, n=1):
    """Return {label: [entries]}: the user's choices, completed with images of the label
    evenly spread over the dataset (consecutive images are often from the same scene)."""
    by_name = {e["name"]: e for e in metadata}
    picked = {}
    for label in labels:
        chosen = []
        for name in examples.get(label, [])[:n]:
            entry = by_name.get(name)
            if entry is None:
                raise ValueError(f"Image '{name}' not found among the plotted images")
            if label not in get_labels(entry):
                print(f"Warning: {name} is not labelled '{label}' (labels: {', '.join(get_labels(entry))})")
            chosen.append(entry)
        candidates = [e for e in metadata if label in get_labels(e) and e not in chosen]
        missing = n - len(chosen)
        if missing > 0 and candidates:
            step = len(candidates) / missing
            chosen += [candidates[int(i * step)] for i in range(min(missing, len(candidates)))]
        picked[label] = chosen
    return picked


def make_mosaic(entries, tile=(240, 176), gap=8):
    """Tile up to 4 images into a 2x2 grayscale square (letterboxed, white gaps)."""
    w, h = tile
    mosaic = np.full((2 * h + gap, 2 * w + gap), 255, dtype=np.uint8)
    for k, entry in enumerate(entries[:4]):
        img = ImageOps.pad(Image.open(image_path(entry)).convert("L"), tile, color=0)
        r, c = divmod(k, 2)
        mosaic[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = np.asarray(img)
    return mosaic


def plot_bars_with_mosaics(totals, labels, examples, n_images, title, ylabel, mosaic_width=95):
    """Vertical bar per label with a 2x2 mosaic of its example images above it."""
    values = [totals[label] for label in labels]

    fig, ax = plt.subplots(figsize=(max(1.45 * len(labels) + 1, 8), 7.5))
    ax.bar(labels, values, color="#4C72B0")
    for i, label in enumerate(labels):
        ax.annotate(f"{values[i]}\n({values[i] / n_images:.0%})", (i, values[i]),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        mosaic = make_mosaic(examples[label])
        ax.add_artist(AnnotationBbox(OffsetImage(mosaic, cmap="gray", norm=plt.Normalize(0, 255),
                                                 zoom=mosaic_width / mosaic.shape[1]),
                                     (i, values[i]), xybox=(0, 30), xycoords="data",
                                     boxcoords="offset points", box_alignment=(0.5, 0), pad=0.2))
        # Image stems, laid out like the mosaic, so they can be reused with --example / --multi-example
        stems = [e["stem"] for e in examples[label]]
        ax.annotate("\n".join(" ".join(stems[k:k + 2]) for k in (0, 2)), (i, values[i]),
                    xytext=(0, 34 + mosaic_width * mosaic.shape[0] / mosaic.shape[1]),
                    textcoords="offset points", ha="center", va="bottom", fontsize=6, color="0.4")
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, max(values) * 1.6)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45 if len(labels) > 8 else 0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title)
    ax.text(0.99, 0.98, "Images above the bars: extracts from the dataset", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, style="italic", color="0.3")
    fig.tight_layout()
    return fig


def plot_repartition(metadata, labels, examples, modality):
    totals = Counter(e["unique_label"] for e in metadata)
    return plot_bars_with_mosaics(totals, labels, examples, len(metadata),
                                  f"Repartition of unique_label - {modality} ({len(metadata)} images)",
                                  "Number of images")


def plot_multi_repartition(metadata, labels, examples, modality):
    totals = Counter(label for e in metadata for label in multi_labels(e))
    return plot_bars_with_mosaics(totals, labels, examples, len(metadata),
                                  f"Repartition of multi_labels - {modality} ({len(metadata)} images, "
                                  "an image counts once per label)",
                                  "Number of images containing the label")


def sort_images(metadata, out_dir):
    for entry in metadata:
        dst = out_dir / entry["unique_label"]
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path(entry), dst / entry["name"])
    print(f"Copied {len(metadata)} images into {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--example", action="append", metavar="LABEL=IMG1,IMG2,...",
                        help="Up to 4 images to display for a unique label (repeatable). "
                             "Missing ones are picked automatically.")
    parser.add_argument("--multi", action="store_true",
                        help="Also plot the multi_labels repartition with 4 example images per label")
    parser.add_argument("--multi-example", action="append", metavar="LABEL=IMG1,IMG2,...",
                        help="Up to 4 images to display for a multi-label (repeatable). "
                             "Missing ones are picked automatically.")
    parser.add_argument("--modality", default="infrared",
                        help="Modality to plot: infrared, visible or all (default: infrared)")
    parser.add_argument("--no-sort", action="store_true", help="Do not copy images into unique_labels/ folders")
    parser.add_argument("--save-dir", type=Path, help="Save figures here instead of only showing them")
    args = parser.parse_args()

    metadata = json.loads(METADATA.read_text())
    plotted = metadata if args.modality == "all" else [e for e in metadata if e["modality"] == args.modality]
    labels = [label for label, _ in Counter(e["unique_label"] for e in plotted).most_common()]

    examples = pick_examples(plotted, labels, parse_examples(args.example),
                             get_labels=lambda e: [e["unique_label"]], n=4)
    fig = plot_repartition(plotted, labels, examples, args.modality)

    if args.multi:
        multi = [label for label, _ in Counter(l for e in plotted for l in multi_labels(e)).most_common()]
        multi_examples = pick_examples(plotted, multi, parse_examples(args.multi_example),
                                       get_labels=multi_labels, n=4)
        fig_multi = plot_multi_repartition(plotted, multi, multi_examples, args.modality)

    if not args.no_sort:
        sort_images(metadata, OUT_DIR)

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_dir / f"unique_label_repartition_{args.modality}.png", dpi=150)
        if args.multi:
            fig_multi.savefig(args.save_dir / f"multi_label_repartition_{args.modality}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()

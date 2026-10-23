"""Region-of-interest (ROI) analysis of the DeepGaze predictions, per ROI class, RGB vs IR.

For every image of both modalities, DeepGaze predicts a fixation density (with the center bias of
the modality as prior). The ROI metrics of analysis/metrics.py (mass, ratio, density,
normalized_ratio) are computed inside the annotated bounding boxes of each class, and of all the
classes together ("all"). The same metrics are computed on the human saliency map, the raw
fixations and the center bias alone, as references. See Comparator.run_deepgaze_roi and
Saver.save_deepgaze_roi for the outputs.

Usage:
    python Experiment_scripts/Run_experiment_3_roi.py
    python Experiment_scripts/Run_experiment_3_roi.py --model MSDB
    python Experiment_scripts/Run_experiment_3_roi.py --classes 1 2 --class-name 1=car --class-name 2=pedestrian
    python Experiment_scripts/Run_experiment_3_roi.py --max-images 3 --output-dir outputs/expe3_roi/test   # quick CPU test
"""
import sys
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # No display on the cluster.

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))


from deepgaze_pytorch import MSDBDataset

from analysis.saver import Saver
from analysis.comparator import Comparator
from helpers.loader import Loader


# The annotations only store category ids. Names inferred by looking at the annotated boxes;
# override with --class-names / --class-name. Ids without a name are called "class_<id>".
DEFAULT_CLASS_NAMES = {
    1: "car",
    2: "pedestrian",
    3: "cyclist",
    4: "truck",
    5: "bus_van",
    6: "motorcycle",
    7: "animal",
    8: "other_vehicle",
    9: "tram",
}


def main():
    loader = Loader(metadata_json)
    saver = Saver(output_dir)

    visible_df = loader.load_dataframe(visible_path)
    infrared_df = loader.load_dataframe(infrared_path)

    comparator = Comparator(
        metadata_json=metadata_json,
        annotations_json=annotations_json,
        sigma_deg=1.0,
        viewing_distance_cm=70.0,
        screen_width_cm=53.1,
        screen_width_px=1360,
    )

    stems = args.stems
    if stems is None and args.max_images is not None:
        stems = comparator._get_stems(infrared_df, visible_df)[:args.max_images]

    roi_results = comparator.run_deepgaze_roi(
        infrared_df=infrared_df,
        visible_df=visible_df,
        infrared_image_dir=images_dir / "infrared",
        visible_image_dir=images_dir / "visible",
        infrared_centerbias_path=infrared_centerbias_path,
        visible_centerbias_path=visible_centerbias_path,
        model=model,
        msdb_dataset=msdb_dataset,
        classes=args.classes,
        class_names=class_names,
        stems=stems,
        keep_maps=not args.no_visualisations,
    )

    per_class = saver.save_deepgaze_roi(
        results=roi_results,
        output_dir=roi_output_dir,
        visualise=not args.no_visualisations,
        n_visualisations=args.n_visualisations,
    )

    print(f"Results saved in {roi_output_dir}")
    columns = ["modality", "source", "class_name", "n_images", "ratio_mean", "normalized_ratio_mean"]
    print(per_class.loc[per_class["source"] == "deepgaze", columns].to_string(index=False))


def parse_class_names(path, pairs):
    """DEFAULT_CLASS_NAMES, updated with a JSON file {"id": "name"} and then with 'id=name' pairs."""
    names = dict(DEFAULT_CLASS_NAMES)
    if path is not None:
        names.update({int(k): str(v) for k, v in json.loads(Path(path).read_text()).items()})
    for pair in pairs or []:
        class_id, sep, name = pair.partition("=")
        if not sep or not class_id.strip().isdigit() or not name.strip():
            raise ValueError(f"--class-name must be 'id=name', got '{pair}'")
        names[int(class_id)] = name.strip()
    if len(set(names.values())) != len(names) or "all" in names.values():
        raise ValueError(f"Class names must be unique and different from 'all': {names}")
    return names


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ROI analysis of the DeepGaze predictions, per ROI class.")
    parser.add_argument("--model", type=str.upper, choices=["IIE", "MSDB"], default="IIE",
                        help="DeepGaze model to evaluate (default: IIE).")
    parser.add_argument("--msdb-dataset", type=str.upper,
                        choices=["MIT1003", "CAT2000", "COCO_FREEVIEW", "DAEMONS", "FIGRIM"],
                        default=None,
                        help="MSDB only: use the parameters learned for this dataset instead of the "
                             "average over its training datasets (default: average).")
    parser.add_argument("--classes", type=int, nargs="+", default=None, metavar="ID",
                        help="ROI class ids to analyse (default: every class of the annotations).")
    parser.add_argument("--class-names", type=Path, default=None, metavar="JSON",
                        help='JSON file {"<id>": "<name>"} overriding the default class names.')
    parser.add_argument("--class-name", action="append", metavar="ID=NAME",
                        help="Name of one class id (repeatable), e.g. --class-name 1=car.")
    parser.add_argument("--stems", nargs="+", default=None,
                        help="Only analyse these image stems (e.g. 1 2 000003).")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Only analyse the first N stems (quick test run).")
    parser.add_argument("--n-visualisations", type=int, default=None,
                        help="Number of random stems to visualise (default: every stem).")
    parser.add_argument("--no-visualisations", action="store_true",
                        help="Do not save the per-image figures (and do not keep the maps in memory).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: outputs/expe3_roi/<roi | roi_msdb[_dataset]>).")
    args = parser.parse_args()
    if args.msdb_dataset is not None and args.model != "MSDB":
        parser.error("--msdb-dataset only applies to --model MSDB")
    if args.stems is not None and args.max_images is not None:
        parser.error("--stems and --max-images are exclusive")

    model = args.model
    msdb_dataset = None
    roi_folder = "roi"
    if model == "MSDB":
        roi_folder = "roi_msdb"
        if args.msdb_dataset is not None:
            msdb_dataset = getattr(MSDBDataset, args.msdb_dataset)
            roi_folder = f"roi_msdb_{args.msdb_dataset.lower()}"

    try:
        class_names = parse_class_names(args.class_names, args.class_name)
    except ValueError as error:
        parser.error(str(error))

    # Access Paths
    data_dir = ROOT / "DATA_OCCULO_2026" / "2026" / "paired"

    images_dir = data_dir / "images"
    output_dir = ROOT / "outputs"
    output_dir_expe3 = output_dir / "expe3_roi"
    roi_output_dir = args.output_dir if args.output_dir is not None else output_dir_expe3 / roi_folder
    metadata_json = data_dir / "metadata" / "metadata.json"
    annotations_json = data_dir / "metadata" / "annotations.json"
    visible_path = data_dir / "dataframes" / "fixations_visible.csv"
    infrared_path = data_dir / "dataframes" / "fixations_infrared.csv"
    infrared_centerbias_path = data_dir / "centerbias" / "infrared.npy"
    visible_centerbias_path = data_dir / "centerbias" / "visible.npy"

    main()

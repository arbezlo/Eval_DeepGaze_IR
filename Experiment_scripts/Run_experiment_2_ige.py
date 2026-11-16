import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))



from deepgaze_pytorch import MSDBDataset

from analysis.saver import Saver
from analysis.comparator import Comparator
from helpers.loader import Loader





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

    # Gold standard: KDE bandwidth, mixture weights and center bias fitted per modality
    gold_standard = comparator.compute_gold_standard(
        infrared_df=infrared_df,
        visible_df=visible_df,
        seed=0,
        infrared_image_dir=images_dir / "infrared",
        visible_image_dir=images_dir / "visible",
    )

    saver.save_gold_standard_results(
        results=gold_standard,
        output_dir=output_dir_expe2 / "gold_standard",
        n_examples=5,
    )

    # Information gain explained by DeepGaze, against the gold standard computed above
    ige_results = comparator.run_deepgaze_ige(
        infrared_df=infrared_df,
        visible_df=visible_df,
        infrared_image_dir=images_dir / "infrared",
        visible_image_dir=images_dir / "visible",
        infrared_centerbias_path=infrared_centerbias_path,
        visible_centerbias_path=visible_centerbias_path,
        gold_standard=gold_standard,
        model=model,
        msdb_dataset=msdb_dataset,
        centerbias_source=centerbias_source,
    )

    saver.save_deepgaze_ige(
        results=ige_results,
        output_dir=output_dir_expe2 / ige_folder,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Gold standard and information gain explained by DeepGaze.")
    parser.add_argument("--model", type=str.upper, choices=["IIE", "MSDB"], default="IIE",
                        help="DeepGaze model to evaluate (default: IIE).")
    parser.add_argument("--msdb-dataset", type=str.upper,
                        choices=["MIT1003", "CAT2000", "COCO_FREEVIEW", "DAEMONS", "FIGRIM"],
                        default=None,
                        help="MSDB only: use the parameters learned for this dataset instead of the "
                             "average over its training datasets (default: average).")
    parser.add_argument("--centerbias", choices=["gold", "file"], default="gold",
                        help="Center-bias prior given to DeepGaze. 'gold' (default): the gold standard's "
                             "leave-one-image-out center bias, i.e. the IG baseline itself. 'file': the "
                             "fixed maps of <data>/centerbias/{infrared,visible}.npy (results go to a "
                             "folder suffixed with _filecb).")
    args = parser.parse_args()
    if args.msdb_dataset is not None and args.model != "MSDB":
        parser.error("--msdb-dataset only applies to --model MSDB")

    model = args.model
    msdb_dataset = None
    ige_folder = "ige"
    if model == "MSDB":
        ige_folder = "ige_msdb"
        if args.msdb_dataset is not None:
            msdb_dataset = getattr(MSDBDataset, args.msdb_dataset)
            ige_folder = f"ige_msdb_{args.msdb_dataset.lower()}"
    centerbias_source = args.centerbias
    if centerbias_source == "file":
        ige_folder += "_filecb"

    # Access Paths
    data_dir = ROOT / "DATA_OCCULO_2026" / "2026" / "paired"

    images_dir = data_dir / "images"
    output_dir = ROOT / "outputs"
    output_dir_expe2 = output_dir / "expe2_ige"
    metadata_json = data_dir / "metadata" / "metadata.json"
    annotations_json = data_dir / "metadata" / "annotations.json"
    visible_path = data_dir / "dataframes" / "fixations_visible.csv"
    infrared_path = data_dir / "dataframes" / "fixations_infrared.csv"
    # Only used with --centerbias file.
    infrared_centerbias_path = data_dir / "centerbias" / "infrared.npy"
    visible_centerbias_path = data_dir / "centerbias" / "visible.npy"

    main()

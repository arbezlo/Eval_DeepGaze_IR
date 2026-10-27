import sys
from pathlib import Path

ROOT = Path().resolve().parents[0]

if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))



from analysis.saver import Saver
from analysis.comparator import Comparator
from analysis.analyzer import Analyzer
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

    expe1_results = comparator.run_expe1(
        infrared_df=infrared_df,
        visible_df=visible_df,
        n_permutations=1000,
        seed=131,
    )


    output_dir_expe1 = output_dir / "expe1_b" / "comparison"
    expe1_df = saver.save_expe1_results(
        results=expe1_results["IR"], 
        output_dir=output_dir_expe1 / "IR"
    )

    output_dir_expe1 = output_dir / "expe1_b" / "comparison"
    expe1_df = saver.save_expe1_results(
        results=expe1_results["VIS"], 
        output_dir=output_dir_expe1 / "VIS"
    )

if __name__ == "__main__":

    # Access Paths
    data_dir = ROOT / "DATA_OCCULO_2026" / "2026" / "paired" 

    images_dir = data_dir / "images"
    output_dir = ROOT / "outputs" 
    metadata_json = data_dir / "metadata" / "metadata.json"
    annotations_json = data_dir / "metadata" / "annotations.json"
    visible_path = data_dir / "dataframes" / "fixations_visible.csv" 
    infrared_path = data_dir / "dataframes" / "fixations_infrared.csv" 
    infrared_centerbias_path = data_dir / "centerbias" / "infrared.npy"
    visible_centerbias_path = data_dir / "centerbias" / "visible.npy"
    
    main()
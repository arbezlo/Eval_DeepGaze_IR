from pathlib import Path

from helpers.visualizer import Visualizer
from saliency.deepgaze import DeepGaze


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "2026" / "paired"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

INFRARED_FIXATIONS_CSV = (
    DATA_DIR / "dataframes" / "fixations_infrared.csv"
)
VISIBLE_FIXATIONS_CSV = (
    DATA_DIR / "dataframes" / "fixations_visible.csv"
)

VISIBLE_IMAGES_DIR = DATA_DIR / "images" / "visible"
INFRARED_IMAGES_DIR = DATA_DIR / "images" / "infrared"


visualizer = Visualizer(
    infrared_dir=INFRARED_IMAGES_DIR,
    visible_dir=VISIBLE_IMAGES_DIR,
    alpha=0.6,
    show_fixation_ids=True,
)


# Affichage des fixations infrarouges
# visualizer.show_fixations(
#     fixations_df=INFRARED_FIXATIONS_CSV,
#     participant_id=None,
#     image_id="1",
#     save_dir=OUTPUT_DIR / "fixations_infrared",
#     point_size=40.0,
#     modality="infrared",
# )


# # Affichage des fixations visibles
# visualizer.show_fixations(
#     fixations_df=VISIBLE_FIXATIONS_CSV,
#     participant_id=None,
#     image_id=None,
#     save_dir=OUTPUT_DIR / "fixations_visible",
#     point_size=40.0,
#     modality="visible",
# )


# Heatmaps infrarouges
# visualizer.show_heatmaps(
#     fixations_df=INFRARED_FIXATIONS_CSV,
#     image_id=None,
#     save_dir=OUTPUT_DIR / "heatmaps_infrared",
#     sigma=35.0,
#     heatmap_alpha=0.45,
#     cmap="jet",
#     use_duration_weights=False,
#     modality="infrared",
# )


# Heatmaps visibles
# visualizer.show_heatmaps(
#     fixations_df=VISIBLE_FIXATIONS_CSV,
#     image_id=None,
#     save_dir=OUTPUT_DIR / "heatmaps_visible",
#     sigma=35.0,
#     heatmap_alpha=0.45,
#     cmap="jet",
#     use_duration_weights=False,
#     modality="visible",
# )


# Comparaison visible / infrarouge
# visualizer.compare_modalities_heatmaps(
#     visible_fixations_df=VISIBLE_FIXATIONS_CSV,
#     infrared_fixations_df=INFRARED_FIXATIONS_CSV,
#     save_dir=OUTPUT_DIR / "human_modalities_comparison",
#     sigma=35.0,
#     heatmap_alpha=0.45,
#     cmap="jet",
#     use_duration_weights=False,
# )


# Comparaison DeepGaze / fixations humaines

deepgaze_predictor = DeepGaze()

visualizer.compare_deepgaze_human_heatmaps(
    fixations_df=INFRARED_FIXATIONS_CSV,
    deepgaze_predictor=deepgaze_predictor,
    modality="infrared",
    save_dir=OUTPUT_DIR / "deepgaze_human__infrared_comparison",
    channel_mode="auto",
    sigma=35.0,
    heatmap_alpha=0.45,
    cmap="jet",
    use_duration_weights=False,
    normalize_for_display=True,
)
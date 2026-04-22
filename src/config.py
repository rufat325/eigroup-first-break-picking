"""Configuration for assets, paths, and model hyperparameters."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "seismic_data"
CACHE_DIR    = PROJECT_ROOT / "cache"
FIGURES_DIR  = PROJECT_ROOT / "figures"

for d in (DATA_DIR, CACHE_DIR, FIGURES_DIR):
    d.mkdir(exist_ok=True)

ASSETS = {
    "lalor": {
        "hdf5":          DATA_DIR / "Lalor_raw_z_1500ms_norp_geom_v3.hdf5",
        "url":           "https://d3sakqnghgsk6x.cloudfront.net/Lalor_3D/Lalor_raw_z_1500ms_norp_geom_v3.hdf5.xz",
        "index_pkl":     CACHE_DIR / "lalor_shot_index.pkl",
        "patch_dir":     CACHE_DIR / "patches_lalor",
        "manifest_pkl":  CACHE_DIR / "lalor_patch_manifest.pkl",
        "samp_rate_ms":  1.0,
    },
    "halfmile": {
        "hdf5":          DATA_DIR / "Halfmile3D_add_geom_sorted.hdf5",
        "url":           "https://d3sakqnghgsk6x.cloudfront.net/Halfmile_3D/Halfmile3D_add_geom_sorted.hdf5.xz",
        "index_pkl":     CACHE_DIR / "halfmile_shot_index.pkl",
        "patch_dir":     CACHE_DIR / "patches_halfmile",
        "manifest_pkl":  CACHE_DIR / "halfmile_patch_manifest.pkl",
        "samp_rate_ms":  2.0,
    },
    "brunswick": {
        "hdf5":          DATA_DIR / "Brunswick_orig_1500ms_V2.hdf5",
        "url":           "https://d3sakqnghgsk6x.cloudfront.net/Brunswick_3D/Brunswick_orig_1500ms_V2.hdf5.xz",
        "index_pkl":     CACHE_DIR / "brunswick_shot_index.pkl",
        "patch_dir":     CACHE_DIR / "patches_brunswick",
        "manifest_pkl":  CACHE_DIR / "brunswick_patch_manifest.pkl",
        "samp_rate_ms":  2.0,
    },
    "sudbury": {
        "hdf5":          DATA_DIR / "preprocessed_Sudbury3D.hdf",
        "url":           "https://d3sakqnghgsk6x.cloudfront.net/Sudbury_3D/preprocessed_Sudbury3D.hdf.xz",
        "index_pkl":     CACHE_DIR / "sudbury_shot_index.pkl",
        "patch_dir":     CACHE_DIR / "patches_sudbury",
        "manifest_pkl":  CACHE_DIR / "sudbury_patch_manifest.pkl",
        "samp_rate_ms":  2.0,
    },
}

PATCH_WIDTH      = 256
PATCH_HEIGHT     = 1024
PATCH_STRIDE     = 128
MASK_THICKNESS   = 2
MIN_LABELED_FRAC = 0.3

SPLIT_SEED  = 42
TRAIN_FRAC  = 0.85
VAL_FRAC    = 0.075

BATCH_SIZE       = 4
LR_FROM_SCRATCH  = 1e-3
LR_FINETUNE      = 1e-4
DICE_WEIGHT      = 1.0
BCE_POS_WEIGHT   = 50.0
BASE_FILTERS     = 32
EPOCHS_SCRATCH   = 25
EPOCHS_FINETUNE  = 10

MIN_PROB_THRESHOLD = 0.3
INFERENCE_STRIDE   = 128
INFERENCE_BATCH    = 8

MODEL_LALOR       = CACHE_DIR / "unet_lalor.keras"
MODEL_HALFMILE_FT = CACHE_DIR / "unet_halfmile_ft.keras"

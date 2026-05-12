LABEL_NAMES = ["normal", "turbulent", "asymmetric", "blocked"]
NUM_CLASSES  = len(LABEL_NAMES)

SENSOR_FEATURES = [
    "front_corr", "sideL_corr", "sideR_corr",
    "d_front", "d_sideL", "d_sideR",
    "asym", "turbulence", "internal_pressure",
    "vibration", "wind_speed",
]

IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-4
WEIGHT_DECAY = 1e-4

DEFAULT_SENSOR_MODEL = "model.pkl"
DEFAULT_SCALER       = "scaler.pkl"
DEFAULT_CV_MODEL     = "cv_model.pt"

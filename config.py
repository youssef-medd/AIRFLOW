LABEL_NAMES = ["normal", "turbulent", "asymmetric", "blocked"]
NUM_CLASSES  = len(LABEL_NAMES)

SENSOR_FEATURES = [
    "front_corr", "sideL_corr", "sideR_corr",
    "d_front", "d_sideL", "d_sideR",
    "asym", "turbulence", "internal_pressure",
    "vibration", "wind_speed",
]

SENSOR_RANGES = {
    "front_corr":       (-1.0,  1.0),
    "sideL_corr":       (-1.0,  1.0),
    "sideR_corr":       (-1.0,  1.0),
    "d_front":          ( 0.0, 10.0),
    "d_sideL":          ( 0.0, 10.0),
    "d_sideR":          ( 0.0, 10.0),
    "asym":             ( 0.0,  5.0),
    "turbulence":       ( 0.0,  3.0),
    "internal_pressure":( 0.5,  2.0),
    "vibration":        ( 0.0,  1.0),
    "wind_speed":       ( 0.0, 30.0),
}

IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS       = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 5

DEFAULT_SENSOR_MODEL = "model.pkl"
DEFAULT_SCALER       = "scaler.pkl"
DEFAULT_CV_MODEL     = "cv_model.pt"

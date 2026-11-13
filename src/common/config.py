from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / 'data'
OUTPUTS_DIR = ROOT / 'outputs'
ASSETS_DIR = ROOT / 'assets'

CARDS = DATA_DIR / 'tcgdex' / 'cards.jsonl'
SETS = DATA_DIR / 'tcgdex' / 'sets.json'
INDEX_DIR = OUTPUTS_DIR / 'index'
MODELS_DIR = OUTPUTS_DIR / 'models'
DEMO_DIR = OUTPUTS_DIR / 'demo'
EVAL_SCANS_DIR = OUTPUTS_DIR / 'eval_scans'

DETECTOR_MODEL = MODELS_DIR / 'detector_unet512.pt'
PRICE_MODEL = MODELS_DIR / 'price_model.joblib'
PRICE_METRICS = MODELS_DIR / 'price_model_metrics.json'
MULTIPLIER_TABLE = OUTPUTS_DIR / 'tiered_multiplier_100.csv'

EMBED_MODEL = 'facebook/dinov2-base'
CARD_W, CARD_H = 600, 800

TOP_K = 5
POOL_SIZE = 100
OCR_SCALE = 2
NAME_BAND = (0, 110, 30, 570)
NUMBER_BAND = (725, 800, 0, 600)
NAME_BONUS = 0.15
NUMBER_BONUS = (0.0, 0.10, 0.20, 0.30)

PREMIUM_FLOOR = 100.0

BRAND = 'Slabwise'
TAGLINE = 'Kartını tara, değerini öğren.'
LOGO = ASSETS_DIR / 'slabwise_logo.png'

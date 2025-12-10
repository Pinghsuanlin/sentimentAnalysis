# ================================= Configuration File ==============================
# a central repository for hyperparameters and file paths, ensuring easy adjustments and consistency across the project.
import os
from pathlib import Path
from transformers import DistilBertTokenizer

# 1. determine project root
try:
    # Check if a typical Docker environment variable or file exists
    # This is a good way to detect containerization
    if os.environ.get('IN_DOCKER') == 'true' or os.path.exists('/.dockerenv'):
        # If in Docker, the root is the WORKDIR set in Dockerfile
        PROJECT_ROOT = Path("/app")
    else:
        # Local run: Calculate path relative to the config file location
        # config.py is inside src/DistilBERT_base, so go up 3 levels
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # each parent goes up one level to reach the root from src/DistilBERT_base/config.py
except Exception:
    # Fallback for unexpected environments
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 2. deinfe file paths
MODEL_SAVE_PATH = PROJECT_ROOT / "model.bin"
TRAINING_FILE = PROJECT_ROOT / "input" / "imdb.csv"

# 3. define hyperparameters
DEVICE = "cuda" # or "cpu" depending on availability
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.1 # 10% of data for validation
MAX_LEN = 128 # maximum length of input text (in tokens) for BERT model
TRAIN_BATCH_SIZE = 16 # 8
VALID_BATCH_SIZE = 4 # 2
LEARNING_RATE = 3e-5 
EPOCHS = 3 # how many times the model will see the entire training dataset
ACCUMULATION_STEPS = 1 # 2 # 8 (batch) * 4 (accum) = 32 effective batch size
BERT_PATH = "distilbert-base-uncased"


TOKENIZER = DistilBertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)






# ================================= Configuration File ==============================
# a central repository for hyperparameters and file paths, ensuring easy adjustments and consistency across the project.
import transformers
import os

# Get the directory where this config file is located, then go up one level to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = "cuda" # or "cpu" depending on availability
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2
MAX_LEN = 128 #256
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 1
LEARNING_RATE = 3e-5
EPOCHS = 10
ACCUMULATION_STEPS = 2
BERT_PATH = "distilbert-base-uncased"
# BERT_PATH = 'bert-base-uncased'  # use HuggingFace repo # alternative
# BERT_PATH = os.path.join(PROJECT_ROOT, "input", "bert_base_uncased")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "model.bin")
TRAINING_FILE = os.path.join(PROJECT_ROOT, "input", "imdb.csv")
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True, local_files_only=True)






#  End-to-End Sentiment Analysis Pipeline: Fine-Tuning DistilBERT and Docker Deployment (IMDB Reviews)

## 1. Project Summary
This project covers a full-stack Machine Learning pipeline, progressing from a traditional NLP baseline model (TF-IDF + Logistic Regression) to a modern Transformer-based model (DistilBERT) for sentiment classification. The final model is containerized using **Docker** and deployed via a **Flask API** for real-time inference.

### Goal
The project goal is build and deploy a robust sentiment classification service that determines whether an IMDB movie text review is "Positive" or "Negative."

---
---

## 2. Project Architecture & Pipeline

The project follows a three-stage development process:

1.  **Exploration & Baseline:** EDA to know word counts distribution and frequent words appearance. And baseline modeling (TF-IDF + Logistic Regression) to establish performance bound.
2.  **Advanced Modeling:** Fine-tuning a deep learning model (DistilBERT).
3.  **Deployment:** Containerize the solution for universal portability and reliable API serving.

---
---

## 3. Data Source and Tools

### Data

* **Source:** [IMDB Dataset of 50k Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* **Subset Used:** 10,000 samples were used for the baseline model to ensure fast iteration.
* **Target:** Binary Classification (`0` for Negative, `1` for Positive).

### Key Packages Used

* **Data/ML:** `pandas`, `numpy`, `scikit-learn`
* **NLP/Deep Learning:** `transformers`, `torch`, `datasets` (for efficient handling of large text data)
* **Deployment:** `Flask`, `Docker`
* **EDA:** `matplotlib`, `seaborn`, `nltk`
---
---


## 4. Modeling and Results

### 4.1 Exploratory Data Analysis (EDA)

* **Class Balance:** The data followed a near-perfect **50/50 balance** between Positive and Negative classes, but still applied stratified sampling and referred to F1-score/Accuracy as fair metrics.
* **Sequence Length:** Found a long-tail distribution with the **$95^{th}$ percentile word count around $590$ words**.
    * **Decision:** Due to the $512$ token limit of standard BERT architecture, **`MAX_LEN` was set to 128** to strike a balance between training efficiency and information capture.

* **File:** `text_eda.ipynb`

---

### 4.2 Model 1: Baseline (TF-IDF + Logistic Regression)
* **File:** `logRrg_TFIDF.py`

| Technique | Parameter/Reasoning |
| :--- | :--- |
| **Feature Extraction** | **TF-IDF Vectorizer** (`min_df=5`, `ngram_range=(1, 2)`, `stop_words='english'`) | Used to weight words based on frequency and rarity. `min_df=5` filtered out noisy, rare words; bigrams captured negation (e.g., "not good"). |
| **Model** | **Logistic Regression** (`solver='liblinear'`, `random_state=42`) | A robust linear baseline chosen for its speed and performance on sparse, high-dimensional data. |
| **Data Handling** | **Stratified Sampling** (`stratify=y`) | Ensured the training and testing sets maintain the $50/50$ class balance, guaranteeing a reliable evaluation. |

#### Baseline Performance (10k Sample)

| Metric | Negative (0) | Positive (1) | Overall |
| :--- | :--- | :--- | :--- |
| **Precision** | $0.89$ | $0.85$ | - |
| **Recall** | $0.84$ | $0.90$ | - |
| **F1-Score** | $0.86$ | $0.87$ | **$0.87$** |
| **Accuracy** | - | - | **$0.87$** |

*Conclusion:* The baseline model performed exceptionally well, achieving $\mathbf{87\%}$ accuracy, establishing a strong benchmark for the advanced model.

---

### 4.3 Model 2: Advanced (DistilBERT / BERT-Base Fine-tuning)
* **Files:** `src/DistilBERT_base`

| Component | Technology | Purpose | Key Files |
| :--- | :--- | :--- | :--- |
| **Model** | *`DistilBERT`* (Base Uncased) | Fast, light, and high-performing NLP model for sequence classification. | `model.py`, `train.py` |
| **Training** | *`PyTorch`, `HuggingFace Transformers`, `AdamW`* | Custom training loop with advanced scheduler and optimization. | `engine.py`, `train.py` |
| **Serving** | *`Flask API`* | Provides a low-latency REST API endpoint for real-time predictions. | `api.py` |
| **Deployment** | *`Docker`* | Containerizes the application, dependencies, and model for consistent, environment-agnostic deployment. |`Dockerfile`, `config.py`|


* **Hyperparameters:** Key parameters like $\mathbf{MAX\_LEN=128}$, $\mathbf{TRAIN\_BATCH\_SIZE=16}$ (limited by VRAM), and a critical **$\mathbf{LEARNING\_RATE=3\text{e-}5}$** were set.
* **Reasoning:** 
    1. Transformers learn rich contextual embeddings, offering superior performance over traditional feature engineering, especially for complex sentiment nuances.
    2. DistilBERT was chosen over the full BERT-Base model due to its efficiency. It is 40% smaller, 60% faster, and retains approximately 97% of BERT's language understanding capabilities, making it ideal for low-latency, production-level serving in a containerized environment.
* **Key Fine-Tuning Techniques:** 
    1. The fine-tuning process involved adding a simple linear classification head on top of the DistilBERT encoder.
        * Structure: [CLS] Output (768 features) $\rightarrow$ Dropout Layer $\rightarrow$ Linear Layer (Output: 1 Logit) $\rightarrow$ Sigmoid (for probability at inference).
    2. Optimizer: Used *`AdamW`* (Adam with weight decay fix) standard for all Transformer models.
    3. Learning Rate Schedule: Implemented a Linear Scheduler with Warmup to stabilize initial training and ensure robust convergence.
        * The weights of the entire pre-trained DistilBERT layer were unfrozen (`param.requires_grad = True`) to allow the model to fully adapt to the sentiment task, maximizing its classification accuracy.

* **Expected Performance:** Expected F1-Score to exceed $\mathbf{90\%}$.

---
---

## 5. Deployment and MLOps (Flask & Docker)

The final fine-tuned model weights were integrated into a production-ready API:

* **Framework:** **Flask** was used to create a simple `/predict` endpoint.
* **Inference Pipeline:** The API loads the **BERT model and tokenizer once** at startup, accepting review text via POST request and returning the predicted sentiment and confidence score.
* **Containerization:** A **Dockerfile** was used to package the Python environment, dependencies, Flask application (`api.py`), and the trained model (`model.bin`) into a single portable image.
* **Key Path:** All deployment assets were mapped to the internal container path, `/root/docker_data/`, ensuring environment independence.

---
---

## 6. Reflections and Next Steps

### Key Learnings/Hurdles

1. **VRAM Management:** The greatest hurdle was training large models like DistilBERT requires significant GPU VRAM. To deal with such hurdle, I used Gradient Accumulation (`ACCUMULATION_STEPS`) over batches to allow for stable training convergence without hitting VRAM limits. Which was implemented in `engine.py`
2. **Robust Pathing and Configuration:** I designed a robust path-finding logic in `config.py` using Python's `pathlib.Path` and a Docker environment variable (`IN_DOCKER=true`), so that the model could find files correctly no matter where it's running from (local path or /app Docker WORKDIR).
3. **Dependency and Deployment Consistency:** Docker Containerization: The Dockerfile locks the application to the python:3.9-slim base image and explicitly defines all dependencies in requirements.txt. This guarantees that the execution environment is identical from training to production.

---
---

## 7. Deployment Quick Start / Running the API
As this project uses Docker to package the fine-tuned BERT model and Flask API, you could launch the complete, containerized service locally with just two commands:


### Prerequisites

* Docker installed and running on your system.

### Steps to Run

1.  **Build the Docker Image** (Execute this command in the project root directory where the `Dockerfile` resides):

    ```bash
    docker build -t sentiment-bert-api .
    ```

2.  **Run the Container** (This launches the API and maps the container's internal port 5000 to your local machine's port 8080):

    ```bash
    docker run -d -p 8080:5000 --name sentiment_service sentiment-bert-api
    ```

3.  **Test the Endpoint** (Use `curl` or Postman to send a test review):

    ```bash
    curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{"text": "This movie was an absolute masterpiece, the best film I have seen all year!"}'
    ```

    *Expected Output:*
    ```json
    {"confidence": 0.998, "sentiment": "Positive", "model": "bert-base-uncased"}
    ```
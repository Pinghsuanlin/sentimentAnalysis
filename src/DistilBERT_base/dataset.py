# ============================= Define Dataset Class ============================= #
# handle text preprocessing and tokenization (converting text into tensors that BERT can process)

from src.DistilBERT_base import config
import torch

class BERTDataset:
    def __init__(self, reviews, targets):
        self.reviews = reviews
        self.targets = targets # 0 or 1
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    # return the length of the dataset
    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        # process each review individually
        review = str(self.reviews[item])
        # remove weird spaces
        review = " ".join(review.split())
        # convert HTML newlines to spaces
        review = review.replace("<br />", " ")

        # if using DistilBERT tokenizer
        inputs = self.tokenizer(
            review,
            None,
            add_special_tokens=True, # Add [CLS] and [SEP]
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # inputs returned with shape (1, seq_len) because of `return_tensors='pt'`
        ids = inputs['input_ids'].squeeze(0).to(dtype=torch.long)        # shape: (seq_len,) and to correct dtypes
        mask = inputs['attention_mask'].squeeze(0).to(dtype=torch.long)  # shape: (seq_len,)
        target = torch.tensor(self.targets[item], dtype=torch.float).unsqueeze(0)  # shape: (1,)

        return {
            'ids': ids,
            'mask': mask,
            'targets': target
        }
# ============================= Define Model Class ============================= #
# create a custom neural network model by extending nn.Module
from transformers import DistilBertModel
from src.DistilBERT_base import config
import torch.nn as nn

# Create a custom neural network model by extending nn.Module
class DistilBERTBaseUncased(nn.Module):
    def __init__(self):
        # initialize the parent class 
        super().__init__()
        # load pre-trained BERT model (or from HuggingFace)
        self.bert = DistilBertModel.from_pretrained(
            config.BERT_PATH, 
            return_dict=False # to get tuple output instead of dict
        )

        # tells PyTorch: "Don't calculate gradients for these 66M parameters"
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # add dropout layer and output layer
        self.dropout = nn.Dropout(0.3) # randomly zeros 30% of neurons to prevent overfitting
        self.out = nn.Linear(768, 1) # BERT base has hidden size of 768
    
    def forward(self, input_ids, attention_mask):
        sequence_out, = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # use the first token ([CLS] token) representation for classification task
        pooled_output = sequence_out[:, 0]
        output = self.dropout(pooled_output)
        output = self.out(output)
        # map to linear layer
        return output
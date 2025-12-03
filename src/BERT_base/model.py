# ============================= Define Model Class ============================= #
# create a custom neural network model by extending nn.Module

import config
import transformers
import torch.nn as nn

# Create a custom neural network model by extending nn.Module
class BERTBaseUncased(nn.Module):
    def __init__(self):
        # initialize the parent class 
        super(BERTBaseUncased, self).__init__()
        # load pre-trained BERT model (or from HuggingFace)
        self.bert = transformers.BertModel.from_pretrained(
            config.BERT_PATH,
            return_dict=False # to get tuple output instead of dict
        )
        # add dropout layer and output layer
        self.dropout = nn.Dropout(0.3) # randomly zeros 30% of neurons to prevent overfitting
        self.out = nn.Linear(768, 1) # BERT base has hidden size of 768
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        """Perform a forward pass of the model
        Args:
            input_ids: token indices, shape [batch_size, max_len]. Map each word to an integer ID from the tokenizer vocabulary
            attention_mask: binary, tensor of shape [batch_size, max_len]. 1 for real tokens, 0 for padding tokens (to be ignored by the model). \
                Padding ensures uniform input sizes in a batch (avoid bias from varying lengths)
            token_type_ids: tensor of shape [batch_size, max_len]
        Returns:
            output: tensor of shape [batch_size, 1]
            """
        out1, out2 = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        #out1: sequence output, shape:[batch_size, 512, 768]; out2: pooled output, shape:[batch_size, 768]
        output = self.dropout(out2)
        # map to linear layer
        return self.out(output) 
import sys
sys.path.append("..")

from torch import nn
from sentence_transformers import SentenceTransformer

from torch import nn
class PrefixEncoder(nn.Module):
    def __init__(self, encoder, num_prefix_token):
        super().__init__()

        self.encoder = encoder
        self.num_prefix_token = num_prefix_token
        self.config = encoder.config
        self.config.num_prefix_token = max(1,num_prefix_token)
        self.main_input_name = "input_ids"

    def forward(self, input_ids, attention_mask, **kwargs):
        enc_out = self.encoder(input_ids = input_ids,
                                     attention_mask= attention_mask)
        enc_out.last_hidden_state = enc_out.last_hidden_state[:, :self.num_prefix_token, :]
        return enc_out

    @classmethod
    def from_sentenc_checkpoint(cls, checkpoint_path):
        sent_transformer = SentenceTransformer(checkpoint_path)

        model = cls(encoder = sent_transformer._first_module().auto_model, 
                    num_prefix_token = sent_transformer._last_module().num_prefix_token)
        
        tokenizer = sent_transformer.tokenizer
     

        return model, tokenizer
    
    def get_output_embeddings(self):
        """Needed for clean loading in EncDec model"""
        return self.encoder.get_output_embeddings()
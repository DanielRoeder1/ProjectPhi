from modeling_gpt2 import GPT2LMHeadModel, GPT2Config
from transformers import AutoModel, AutoTokenizer
from tokenizers.processors import TemplateProcessing
from torch import nn

logging_steps = 100
batch_size = 16
learning_rate = 1e-4
model_checkpoint = "/netscratch/roeder/phi_train/run_fancy-paper-508/checkpoint-1500"
freeze_decoder = True
cross_attention_layer = [11,10,9]
use_embed_layer = True 

import torch 
class EmbedEncoder(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.wte = decoder.transformer.wte
        self.wpe = decoder.transformer.wpe
    
    def forward(self,input_ids, attention_mask):
        past_length = 0 
        input_shape = input_ids.size()
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=decoder.device)
        position_ids = position_ids.unsqueeze(0)
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        return hidden_states




class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, batch):
        enc_hidden_state = encoder(input_ids = batch["input_ids"], 
                                   attention_mask = batch["attention_mask"]).last_hidden_state
        
        out = decoder(input_ids = batch["decoder_input_ids"], 
                     attention_mask = batch["decoder_attention_mask"],
                     encoder_hidden_states = enc_hidden_state, 
                     encoder_attention_mask = batch["attention_mask"], 
                     labels = batch["labels"])
        return out

def freeze_decoder(model):
    for n,p in model.named_parameters():
        if not "cross" in n:
            p.requires_grad = False

config = GPT2Config.from_pretrained("gpt2")
config.add_cross_attention = True
config.cross_attn_layer_idx = cross_attention_layer

decoder = GPT2LMHeadModel.from_pretrained(model_checkpoint, config = config)

dec_tokenizer = AutoTokenizer.from_pretrained("gpt2")
dec_tokenizer.pad_token = dec_tokenizer.eos_token
dec_tokenizer._tokenizer.post_processor = TemplateProcessing(
    single= dec_tokenizer.bos_token + " $A " + dec_tokenizer.eos_token,
    special_tokens=[(dec_tokenizer.bos_token, dec_tokenizer.bos_token_id),(dec_tokenizer.eos_token, dec_tokenizer.eos_token_id)],
)

if use_embed_layer:
    encoder = EmbedEncoder(decoder)
    enc_tokenizer = dec_tokenizer
else:
    encoder = AutoModel.from_pretrained("roberta-base")
    enc_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

if freeze_decoder:
    freeze_decoder(decoder)

model  = EncoderDecoder(encoder = encoder,decoder = decoder)
model = model.to("cuda")

from train_utils.utils import prepare_dataset, prompt_qc, prompt_q, prompt_qa
from datasets import load_dataset

dataset = load_dataset("squad")
dataset["train"] = dataset["train"].select(range(1500, len(dataset["train"])))
dataset = dataset.map(lambda x: {"answers": x["answers"]["text"][0]})
dataset = dataset.map(lambda x: {k:v.strip() for k,v in x.items()})

df_qa = dataset.map(prepare_dataset, 
                     fn_kwargs={"prompt": prompt_qa, 
                                "tokenizer": dec_tokenizer, 
                                "create_labels" : True, 
                                "enc_tokenizer": enc_tokenizer, 
                                "context_enc": True,
                                "enc_prompt": prompt_qc}, 
                     batched=True, 
                     remove_columns=dataset["train"].column_names)

from train_utils.utils import CustomCollator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

optimizer = AdamW(model.parameters(), lr = learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 500, num_training_steps = df_qa["train"].num_rows // batch_size)
train_loader = DataLoader(df_qa["train"],batch_size =batch_size, collate_fn=CustomCollator(dec_tokenizer, enc_tokenizer))

import wandb

wandb.login(key = "f190694cef6354f5205256582202a2b16502a236")
wandb.init()
wandb.watch(model, log="all")

total_loss = 0

for i, batch in tqdm(enumerate(train_loader)):
    batch = {k:v.to(model.decoder.device) for k,v in batch.items()}
    optimizer.zero_grad()
    out = model(batch)
    out.loss.backward()
    optimizer.step()
    scheduler.step()

    total_loss += out.loss.item()
    
    if (i+1) % logging_steps == 0:
        print(f"Loss: {total_loss/logging_steps}")
        wandb.log({"loss": total_loss/logging_steps, "lr": scheduler.get_last_lr()})
        total_loss = 0
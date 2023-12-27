from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, EncoderDecoderConfig
from omegaconf import OmegaConf
import os
import sys
sys.path.append("..")

from train_utils.EncoderDecoder import CustomEncoderDecoderModel
from train_utils.encoder import PrefixEncoder

def load_encdec_configs(checkpoint_path):
    encdec_config = AutoConfig.from_pretrained(checkpoint_path)
    enc_config = encdec_config.encoder
    dec_config = encdec_config.decoder
    return enc_config, dec_config

def load_encdec_statedict(checkpoint_path):
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    tensors = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k,) # loads the full tensor given a key
    enc_state_dict ={k.replace("encoder.","",1):v for k,v in tensors.items() if "encoder" in k}
    dec_state_dict ={k.replace("decoder.","",1):v for k,v in tensors.items() if "decoder" in k}
    encdec_proj_dict = {k.replace("enc_to_dec_proj.",""):v for k,v in tensors.items() if "enc_to_dec_proj" in k}
    return enc_state_dict, dec_state_dict, encdec_proj_dict


def load_encdec_model(checkpoint_path, enc_model_class, dec_model_class):
    train_conf = OmegaConf.load(os.path.join(checkpoint_path, "model_config.yaml"))

    enc_state_dict, dec_state_dict, encdec_proj_dict = load_encdec_statedict(checkpoint_path)
    enc_config, dec_config = load_encdec_configs(checkpoint_path)
    
    if hasattr(enc_config, "num_prefix_token"):
        enc_model = PrefixEncoder(encoder = enc_model_class.from_config(enc_config), 
                                  num_prefix_token = enc_config.num_prefix_token)
        enc_model.load_state_dict(enc_state_dict)
    else:
        enc_model = enc_model_class.from_pretrained(enc_config._name_or_path, config = enc_config, state_dict = enc_state_dict)
    
    enc_tokenizer_path = os.path.join(checkpoint_path, "enc_tokenizer")
    if os.path.isdir(enc_tokenizer_path):
        enc_tokenizer = AutoTokenizer.from_pretrained(enc_tokenizer_path)
    else:
        enc_tokenizer = AutoTokenizer.from_pretrained(enc_config._name_or_path)

    dec_model = dec_model_class.from_pretrained(checkpoint_path, config = dec_config, state_dict = dec_state_dict)
    dec_tokenizer = AutoTokenizer.from_pretrained(train_conf.model_args.decoder_base_name)

    config = EncoderDecoderConfig(**{"encoder": enc_config.to_dict(), "decoder": dec_config.to_dict()})
    model = CustomEncoderDecoderModel(encoder=enc_model, decoder=dec_model, config = config)
    if len(encdec_proj_dict) > 0:
        model.enc_to_dec_proj.load_state_dict(encdec_proj_dict)
    return model, enc_tokenizer, dec_tokenizer, train_conf

import torch
from collections import defaultdict
import numpy as np
 
def load_batches_from_evaldf(df, batch_id = 0):
    dec_loss_batch, dec_gen_batch = None, None
    # list_of_dicts -> dic_of_lists
    def transpose_dict(list_of_dicts):
        dict_of_lists = defaultdict(list)

        for d in list_of_dicts:
            for key, value in d.items():
                dict_of_lists[key].append(value)

        dict_of_lists = dict(dict_of_lists)
        return dict_of_lists

    df_select = df[df.batch_id == batch_id]
    loss_batch = transpose_dict(df_select.loss_batch)
    gen_batch = transpose_dict(df_select.gen_batch)
    loss_batch = {k:torch.tensor(np.array(v)) for k,v in loss_batch.items()}
    gen_batch = {k:torch.tensor(np.array(v)) for k,v in gen_batch.items()}
    if "decoder_input_ids" in loss_batch:
        dec_loss_batch = {k.replace("decoder_",""):v for k,v in loss_batch.items() if k in ["decoder_input_ids", "decoder_attention_mask", "labels"]}
        dec_gen_batch = {k.replace("decoder_",""):v for k,v in gen_batch.items() if k in ["decoder_input_ids", "decoder_attention_mask", "labels"]}

            
    return loss_batch, gen_batch, dec_loss_batch, dec_gen_batch



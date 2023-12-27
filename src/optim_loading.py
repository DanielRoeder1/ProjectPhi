def prepare_model_parameters(model):
    encoder_nodecay = [p for n,p in model.encoder.named_parameters() if any([x in n for x in ["bias", "LayerNorm"]])]
    encoder_decay = [p for n,p in model.encoder.named_parameters() if not any([x in n for x in ["bias", "LayerNorm"]])]

    decoder_cross_nodecay = [p for n,p in model.decoder.named_parameters() if any([x in n for x in ["cross","attn_gate"]]) and any([x in n for x in ["bias", "ln"]])]
    decoder_cross_decay = [p for n,p in model.decoder.named_parameters() if any([x in n for x in ["cross","attn_gate"]]) and not any([x in n for x in ["bias", "ln"]])]

    decoder_backbone_nodecay = [p for n,p in model.decoder.named_parameters() if not any([x in n for x in ["cross","attn_gate"]]) and any([x in n for x in ["bias", "ln"]])]
    decoder_backbone_decay = [p for n,p in model.decoder.named_parameters() if not any([x in n for x in ["cross","attn_gate"]]) and not any([x in n for x in ["bias", "ln"]])]

    if hasattr(model, "enc_to_dec_proj"):
        decoder_cross_nodecay += [model.enc_to_dec_proj.bias]
        decoder_cross_decay += [model.enc_to_dec_proj.weight]
    return {"decoder_backbone_decay": decoder_backbone_decay,
            "decoder_backbone_nodecay": decoder_backbone_nodecay,
            "decoder_cross_decay": decoder_cross_decay,
            "decoder_cross_nodecay": decoder_cross_nodecay,
            "encoder_decay": encoder_decay,
            "encoder_nodecay": encoder_nodecay}


def get_group_params(train_type:str, param_dict):    
    params_decay = []
    params_nodecay = []
    if "enc" in train_type or train_type == "full":
        params_decay+= param_dict.pop("encoder_decay") if "encoder_decay" in param_dict else []
        params_nodecay+= param_dict.pop("encoder_nodecay") if "encoder_nodecay" in param_dict else []
    if "cross" in train_type or train_type == "full":
        params_decay += param_dict.pop("decoder_cross_decay") if "decoder_cross_decay" in param_dict else []
        params_nodecay += param_dict.pop("decoder_cross_nodecay") if "decoder_cross_nodecay" in param_dict else []
    if "dec" in train_type or train_type == "full":
        params_decay += param_dict.pop("decoder_backbone_decay") if "decoder_backbone_decay" in param_dict else []
        params_nodecay += param_dict.pop("decoder_backbone_nodecay") if "decoder_backbone_nodecay" in param_dict else []
    return params_decay, params_nodecay

def get_g1_g2(model, train_type1, train_type2):
    all_param_groups = prepare_model_parameters(model)
    g1_decay_params, g1_nodecay_params = get_group_params(train_type1, all_param_groups)
    g2_decay_params, g2_nodecay_params = get_group_params(train_type2, all_param_groups)
    
    return g1_decay_params, g1_nodecay_params, g2_decay_params, g2_nodecay_params

from torch.optim import AdamW
from optim_scheduler import DifferentialAlignmentSchedulerWithZeroPeriodLRFixed

def get_optimizer_and_scheduler(optim_args,g1_decay_params, g1_nodecay_params, g2_decay_params, g2_nodecay_params):
    dummy_lr = 1e-9
    optimizer_grouped_parameters = [
        {
            "params": g1_decay_params,
            "weight_decay": optim_args.weight_decay,
            'lr': dummy_lr
        },
        {
            "params": g1_nodecay_params,
            "weight_decay": 0.0,
            'lr': dummy_lr
        },
        {
            "params": g2_decay_params,
            "weight_decay": optim_args.weight_decay,
            'lr': dummy_lr
        },
        {
            "params": g2_nodecay_params,
            "weight_decay": 0.0,
            'lr': dummy_lr
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    scheduler = DifferentialAlignmentSchedulerWithZeroPeriodLRFixed(optimizer, 
                                                                    warmup_steps_g1= optim_args.warmup_steps_g1, 
                                                                    warmup_steps_g2= optim_args.warmup_steps_g2, 
                                                                    lr_g1= optim_args.lr_g1,
                                                                    total_steps=optim_args.total_steps, 
                                                                    zero_period_steps = optim_args.zero_period_steps, 
                                                                    zero_period_lr = optim_args.zero_period_lr,
                                                                    decay = optim_args.decay_type)
    return optimizer, scheduler
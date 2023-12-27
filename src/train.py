from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer,AutoConfig,TrainingArguments
from tokenizers.processors import TemplateProcessing
import wandb
from omegaconf import OmegaConf
import os

from transformers import EncoderDecoderConfig, AutoModel
from eval_utils.loading_utils import load_encdec_model
from modeling_gpt2 import GPT2LMHeadModel

from train_utils.utils import prepare_dataset, CustomCollator, prompt_qca, prompt_qa, prompt_article, prompt_article_summary, prompt_qc_enc
from train_utils.eval import evaluate
from train_utils.Trainer import CustomTrainer, UpdateOutputDirCallback, AdditionalEvalCallback


wandb.login(key = "f190694cef6354f5205256582202a2b16502a236")
args = OmegaConf.load("configs/default.yaml")


train_enc_dec = args.model_args.is_enc_dec
freeze_decoder = args.model_args.freeze_decoder
freeze_encoder = args.model_args.freeze_encoder
create_labels = args.data_args.cover_labels
encoder_model = args.model_args.encoder_name
decoder_checkpoint = args.model_args.decoder_name
if args.data_args.prompt_type == "qc":
    prompt = prompt_qca
elif args.data_args.prompt_type == "q":
    prompt = prompt_qa
elif args.data_args.prompt_type == "article":
    prompt = prompt_article
elif args.data_args.prompt_type == "article_summary":
    prompt = prompt_article_summary


config = AutoConfig.from_pretrained(args.model_args.decoder_base_name, trust_remote_code = True)
config.know_type = args.model_args.adapter_args.adapter_type
config.enc_dim = args.model_args.adapter_args.enc_dim
config.know_layer = OmegaConf.to_container(args.model_args.adapter_args.know_layer)
config.hidden_dropout = args.model_args.adapter_args.hidden_dropout
config.know_proj_bias = args.model_args.adapter_args.proj_bias
config.know_pos = args.model_args.adapter_args.know_pos
config.know_norm = args.model_args.adapter_args.know_norm



tokenizer = AutoTokenizer.from_pretrained(args.model_args.decoder_base_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer._tokenizer.post_processor = TemplateProcessing(
    single= tokenizer.bos_token + " $A " + tokenizer.eos_token,
    special_tokens=[(tokenizer.bos_token, tokenizer.bos_token_id),(tokenizer.eos_token, tokenizer.eos_token_id)],
)

if os.path.isdir(decoder_checkpoint) and isinstance((conf:=AutoConfig.from_pretrained(decoder_checkpoint)), EncoderDecoderConfig):
    print("---- Loading Encoder Decoder Model ----	")
    model, enc_tokenizer, dec_tokenizer, train_conf = load_encdec_model(decoder_checkpoint, enc_model_class= AutoModel, dec_model_class= GPT2LMHeadModel)
else:
    if "phi" in args.model_args.decoder_base_name:
        from phi.modeling_phi import PhiForCausalLM
        model = PhiForCausalLM.from_pretrained(decoder_checkpoint, config=config)
    elif "gpt2" in args.model_args.decoder_base_name:
        from modeling_gpt2 import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(decoder_checkpoint, config=config)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(decoder_checkpoint, config=config)

    if train_enc_dec:
        from transformers import EncoderDecoderConfig, AutoModel, AutoConfig, AutoTokenizer
        from train_utils.EncoderDecoder import CustomEncoderDecoderModel

        if os.path.isdir(encoder_model) and "config_sentence_transformers.json" in os.listdir(encoder_model):
            print("---- loading Sentence Transformer Encoder ----")
            from train_utils.encoder import PrefixEncoder
            enc_model, enc_tokenizer = PrefixEncoder.from_sentenc_checkpoint(encoder_model)
        else:
            enc_model = AutoModel.from_pretrained(encoder_model)
            enc_tokenizer = AutoTokenizer.from_pretrained(encoder_model)


        config = EncoderDecoderConfig(**{"encoder": enc_model.config.to_dict(), "decoder": AutoConfig.from_pretrained("gpt2").to_dict()})
        config.decoder= model.config
        model = CustomEncoderDecoderModel(encoder=enc_model, decoder=model, config = config)
    else:
        enc_tokenizer = None

if freeze_encoder:
    for n,p in model.encoder.named_parameters():
        p.requires_grad = False

if freeze_decoder:
    for n, p in (model.decoder if hasattr(model, "decoder") else model).named_parameters():
        if not "proj_k" in n and not "proj_v" in n and "gated_attn" not in n and "cross" not in n:
            p.requires_grad = False

model.args = args

dataset = load_from_disk(args.data_args.dataset_path) if os.path.isdir(args.data_args.dataset_path) else load_dataset(args.data_args.dataset_path)
if "squad" in args.data_args.dataset_path:
    if args.data_args.context_column == "answer_sentence" and "answer_sentence" not in dataset["train"].column_names:
        from train_utils.utils import extract_sentence
        dataset = dataset.map(lambda x: {"answer_sentence": extract_sentence(x["context"], x["answers"]["answer_start"][0])})

    dataset = dataset.map(lambda x: {"answers": x["answers"]["text"][0]})
    dataset = dataset.map(lambda x: {k:v.strip() for k,v in x.items()})
elif "cnn" in args.data_args.dataset_path:
    column_names = dataset["train"].column_names
    dataset = dataset.map(lambda x: {"summary": [" ".join(entry.split()[:100]) for entry in x["highlights"]]}, batched=True)
    dataset = dataset.map(lambda x: {"article": [" ".join(entry.split()[:200]) for entry in x["article_half"]]}, batched=True, remove_columns=column_names)

df_qca = dataset.map(prepare_dataset, 
                     fn_kwargs={"prompt": prompt, 
                                "tokenizer": tokenizer, 
                                "create_labels" : create_labels, 
                                "enc_tokenizer": enc_tokenizer, 
                                "context_enc": train_enc_dec, 
                                "context_column": args.data_args.context_column,
                                "answer_column": args.data_args.answer_column,
                                "enc_prompt": prompt_qc_enc if args.data_args.context_column == "qc" else None,
                                "num_prefix_token": model.encoder.num_prefix_token if train_enc_dec and hasattr(model.encoder, "num_prefix_token") else 0}, 
                     batched=True, 
                     remove_columns=dataset["train"].column_names)

if "cnn" in args.data_args.dataset_path:
    # Indices resulting in really long input sequences
    indices_to_drop = [60486, 69092, 98277, 157444, 173621]

    def filter_indices(row, index):
        return index not in indices_to_drop

    df_qca = df_qca.filter(filter_indices, with_indices=True)

custom_optim = args.model_args.custom_optim
if custom_optim:
    from optim_loading import get_g1_g2, get_optimizer_and_scheduler
    from types import SimpleNamespace
    g1_decay_params, g1_nodecay_params, g2_decay_params, g2_nodecay_params = get_g1_g2(model, args.model_args.train_type1, args.model_args.train_type2)

    optim_args = SimpleNamespace(weight_decay = 0.01, 
                                warmup_steps_g1= 500, 
                                warmup_steps_g2= 500, 
                                zero_period_steps= 5475, 
                                total_steps= 10950, 
                                lr_g1= args.training_args.learning_rate, 
                                zero_period_lr= 0,
                                decay_type = "linear")

    
    optimizer, scheduler = get_optimizer_and_scheduler(optim_args,g1_decay_params, g1_nodecay_params, g2_decay_params, g2_nodecay_params)

train_default = False

if train_default:
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    import wandb

    wandb.init()
    wandb.watch(model, log="all")

    optimizer = AdamW(model.parameters(), lr = 1e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 500, num_training_steps = 5475)
    collate_fn = CustomCollator(tokenizer, enc_tokenizer = enc_tokenizer)
    train_loader = DataLoader(df_qca["train"], batch_size = 4, collate_fn = collate_fn)


    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        batch = {k:v.to(model.device) for k,v in batch.items()}
        optimizer.zero_grad()
        out = model(**batch)
        out.loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += out.loss.item()

        if i % 100 == 0:
            print(f"Loss: {running_loss/(i+1)}")
            wandb.log({"loss": running_loss/(i+1)})
else:

    training_args = TrainingArguments(**args.training_args)

    # Save scripts along checkpoint
    if len(args.extra_args.script_paths) > 0:
        training_args.script_paths = args.extra_args.script_paths
    training_args.watch_wandb = args.extra_args.watch_wandb

    trainer = CustomTrainer(
        model=model,
        args= training_args,
        train_dataset=df_qca["train"],
        eval_dataset=df_qca["validation"],
        tokenizer=tokenizer,
        enc_tokenizer = enc_tokenizer,
        data_collator=CustomCollator(tokenizer, enc_tokenizer = enc_tokenizer),
        optimizers = (optimizer, scheduler) if custom_optim else (None, None),
        callbacks=[UpdateOutputDirCallback()],
    )
    trainer.train()

    if train_enc_dec:
        print("-------------------- Training Samples --------------------")
        loader = trainer.get_train_dataloader()
        #labels = "".join(tokenizer.batch_decode(batch["labels"][0]))
        for i, batch in enumerate(loader):
            enc_input = "".join(enc_tokenizer.batch_decode(batch["input_ids"][0]))
            dec_input = "".join(tokenizer.batch_decode(batch["decoder_input_ids"][0]))
            print(f"Encoder Input: {enc_input}")
            print(f"Deocder Input: {dec_input}")
            print(f"Labels Input: {batch['labels'][0]}")
            print(f"Loss: {trainer.model(**batch).loss}")
            print("########################")
            if i == 4: break

        print("-------------------- Validation Samples --------------------")
        loader = trainer.get_eval_dataloader()
        #labels = "".join(tokenizer.batch_decode(batch["labels"][0]))
        for i, batch in enumerate(loader):
            enc_input = "".join(enc_tokenizer.batch_decode(batch["input_ids"][0]))
            dec_input = "".join(tokenizer.batch_decode(batch["decoder_input_ids"][0]))
            print(f"Encoder Input: {enc_input}")
            print(f"Deocder Input: {dec_input}")
            print(f"Labels Input: {batch['labels'][0]}")
            print(f"Loss: {trainer.model(**batch).loss}")
            print("########################")
            if i == 4: break

    eval_output = evaluate(trainer.model,
                           tokenizer, 
                           enc_tokenizer, 
                           dataset_path = args.data_args.dataset_path,
                           prompt_type = args.data_args.prompt_type, 
                           context_enc = train_enc_dec, 
                           cover_labels = create_labels, 
                           context_column = args.data_args.context_column,
                           run_decoder_only = train_enc_dec,
                           num_prefix_token = model.encoder.num_prefix_token if train_enc_dec and hasattr(model.encoder, "num_prefix_token") else 0
                          )

    save_path = os.path.join(trainer.args.output_dir, "eval_output.pkl")
    eval_output.to_pickle(save_path)
    bert_score = eval_output.bert_score.apply(lambda x: x["f1"][0]).mean()
    bleu = eval_output.bleu.apply(lambda x: x["bleu"]).mean()
    rougel = eval_output.rouge.apply(lambda x: x["rougeL"]).mean()
    rouge1 = eval_output.rouge.apply(lambda x: x["rouge1"]).mean()
    rouge2 = eval_output.rouge.apply(lambda x: x["rouge2"]).mean()

    print(f"Exact Match: {eval_output.exact_match.mean()}")
    print(f"F1: {eval_output.f1.mean()}")
    print(f"Bert Score: {bert_score}")
    print(f"Bleu: {bleu}")
    print(f"Rouge L: {rougel}")
    print(f"Rouge 1: {rouge1}")
    print(f"Rouge 2: {rouge2}")
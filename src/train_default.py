from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer,AutoConfig,TrainingArguments
from tokenizers.processors import TemplateProcessing
import wandb
from omegaconf import OmegaConf
import os

from phi.modeling_phi import PhiForCausalLM
from train_utils.utils import prepare_dataset, CustomCollator, prompt_qca, prompt_qa
from train_utils.eval import evaluate
from train_utils.Trainer import CustomTrainer, UpdateOutputDirCallback


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


config = AutoConfig.from_pretrained("microsoft/phi-1_5", trust_remote_code = True)
config.know_type = args.model_args.adapter_args.adapter_type
config.enc_dim = args.model_args.adapter_args.enc_dim
config.know_layer = OmegaConf.to_container(args.model_args.adapter_args.know_layer)
config.hidden_dropout = args.model_args.adapter_args.hidden_dropout
config.know_proj_bias = args.model_args.adapter_args.proj_bias

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
tokenizer.pad_token = tokenizer.eos_token
tokenizer._tokenizer.post_processor = TemplateProcessing(
    single= tokenizer.bos_token + " $A " + tokenizer.eos_token,
    special_tokens=[(tokenizer.bos_token, tokenizer.bos_token_id),(tokenizer.eos_token, tokenizer.eos_token_id)],
)
model = PhiForCausalLM.from_pretrained(decoder_checkpoint, config=config)

if freeze_decoder:
    for n, p in model.named_parameters():
        if not "proj_k" in n and not "proj_v" in n and "gated_attn" not in n:
            p.requires_grad = False

if train_enc_dec:
    from transformers import EncoderDecoderConfig, AutoModel, AutoConfig, AutoTokenizer
    from train_utils.EncoderDecoder import CustomEncoderDecoderModel

    enc_model = AutoModel.from_pretrained(encoder_model)
    enc_tokenizer = AutoTokenizer.from_pretrained(encoder_model)

    if freeze_encoder:
        for n,p in enc_model.named_parameters():
            p.requires_grad = False

    config = EncoderDecoderConfig(**{"encoder": enc_model.config.to_dict(), "decoder": AutoConfig.from_pretrained("gpt2").to_dict()})
    config.decoder= model.config
    model = CustomEncoderDecoderModel(encoder=enc_model, decoder=model, config = config)
else:
    enc_tokenizer = None

model.args = args

dataset = load_from_disk(args.data_args.dataset_path) if os.path.isdir(args.data_args.dataset_path) else load_dataset(args.data_args.dataset_path)
if args.data_args.context_column == "answer_sentence" and "answer_sentence" not in dataset["train"].column_names:
    from train_utils.utils import extract_sentence
    dataset = dataset.map(lambda x: {"answer_sentence": extract_sentence(x["context"], x["answers"]["answer_start"][0])})

dataset = dataset.map(lambda x: {"answers": x["answers"]["text"][0]})
df_qca = dataset.map(prepare_dataset, 
                     fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "create_labels" : create_labels, "enc_tokenizer": enc_tokenizer, "context_enc": train_enc_dec, "context_column": args.data_args.context_column}, 
                     batched=True, 
                     remove_columns=dataset["train"].column_names)


train_default = True

if train_default:
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    import wandb

    wandb.init()
    wandb.watch(model, log="all")

    optimizer = AdamW(model.parameters(), lr = 1e-4).zero_grad()
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
        data_collator=CustomCollator(tokenizer, enc_tokenizer = enc_tokenizer),
        callbacks=[UpdateOutputDirCallback()],
    )
    trainer.train()


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
                        run_decoder_only = True)

    save_path = os.path.join(trainer.args.output_dir, "eval_output.pkl")
    eval_output.to_pickle(save_path)
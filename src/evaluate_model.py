from eval_utils.loading_utils import load_encdec_model
from modeling_gpt2 import GPT2LMHeadModel
from transformers import AutoModel
from train_utils.eval import evaluate
from omegaconf import OmegaConf
import os


CHECKPOINT_PATH = "/netscratch/roeder/phi_train/run_spring-pond-589/checkpoint-10950"
run = CHECKPOINT_PATH.split("/")[-2]

args = OmegaConf.load(os.path.join(CHECKPOINT_PATH, "model_config.yaml"))

model, enc_tokenizer, dec_tokenizer, train_conf = load_encdec_model(CHECKPOINT_PATH, AutoModel, GPT2LMHeadModel)
model = model.to("cuda")
eval_output = evaluate(model = model, 
                        tokenizer = dec_tokenizer, 
                        enc_tokenizer=enc_tokenizer, 
                        dataset_path=args.data_args.dataset_path, 
                        prompt_type=args.data_args.prompt_type, 
                        context_enc=args.model_args.is_enc_dec,
                        cover_labels=args.data_args.cover_labels, 
                        batch_size=4, 
                        context_column=args.data_args.context_column, 
                        answer_column=args.data_args.answer_column, 
                        run_decoder_only=False,
                        save_logits=False,
                        num_prefix_token= model.encoder.num_prefix_token if hasattr(model, "encoder") and hasattr(model.encoder, "num_prefix_token") else 0,
                       )

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

eval_output.to_pickle(f"/netscratch/roeder/phi_train/eval_results_{run}.pkl")
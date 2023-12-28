from torch.utils.data import DataLoader
import pandas as pd
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from tokenizers.processors import TemplateProcessing
import torch
from torch.nn.utils.rnn import pad_sequence
import os
from evaluate import load

from .metrics import exact, solution_present, f1
from .utils import prompt_qc, prompt_qca, prompt_q, prompt_qa, prompt_article, prompt_article_summary, prompt_article_summary_gen, prompt_article_gen, prompt_qc_enc

class EvalCollator:
    """Creates one batch for generation and one batch for loss calculation
       Generation batch: no answer included, only eos
       Loss batch: includes answer, both bos and eos
       Expects dataset columns: context, question, answers"""
    def __init__(self, tokenizer, enc_tokenizer, mode = "qc", context_enc = False, cover_labels = False, context_column = "context", answer_column = "answers", num_prefix_token = 0) -> None:
        self.dec_tokenizer = tokenizer
        self.enc_tokenizer = enc_tokenizer
        self.context_enc = context_enc
        self.cover_labels = cover_labels
        self.context_column = context_column
        self.answer_column = answer_column
        self.context_prompt = prompt_qc_enc if context_column == "qc" else None

        if num_prefix_token -1 > 0:
            self.context_prompt = "".join([f"<PREFIX{i}>" for i in range(num_prefix_token-1)]) +  self.context_prompt

        if mode == "qc":
            self.gen_prompt = prompt_qc
            self.loss_prompt = prompt_qca
        elif mode == "q":
            self.gen_prompt = prompt_q
            self.loss_prompt = prompt_qa
        elif mode == "article":
            self.gen_prompt = prompt_article_gen
            self.loss_prompt = prompt_article
        elif mode == "article_summary":
            self.gen_prompt = prompt_article_summary_gen
            self.loss_prompt = prompt_article_summary
        else:
            raise ValueError("mode most be one of [qc,q, article, article_summary]")

        self.loss_processor = TemplateProcessing(
            single= tokenizer.bos_token + " $A " + tokenizer.eos_token,
            special_tokens=[(tokenizer.bos_token, tokenizer.bos_token_id),(tokenizer.eos_token, tokenizer.eos_token_id)],
        )
        self.gen_processor = TemplateProcessing(
            single= tokenizer.bos_token + " $A",
            special_tokens=[(tokenizer.bos_token, tokenizer.bos_token_id)],
        )
        
        self.IGNORE_INDEX = -100
    
        assert not (mode == "qc" and context_enc), "If mode is qc then no context encoder is utilized. Set context_enc to False"
    
    def __call__(self, batch):
        self.dec_tokenizer.padding_side = "left"
        self.dec_tokenizer._tokenizer.post_processor = self.gen_processor
        gen_tokens = self.dec_tokenizer([self.gen_prompt.format(**sample) for sample in batch], return_tensors = "pt", padding = True)
        
        self.dec_tokenizer.padding_side = "right"
        self.dec_tokenizer._tokenizer.post_processor = self.loss_processor
        loss_text = [self.loss_prompt.format(**sample) for sample in batch]
        loss_tokens = self.dec_tokenizer(loss_text, return_tensors = "pt", padding = True)

        label_tokens = self.dec_tokenizer(loss_text, padding = False)["input_ids"]
        if self.cover_labels:
            answer_tokens = self.dec_tokenizer([sample[self.answer_column] for sample in batch], padding = False)["input_ids"]
            label_tokens = [(len(l_t)+1-len(a_t))*[-100] + l_t[-len(a_t)+1:] for l_t, a_t in zip(label_tokens, answer_tokens)]
        labels = pad_sequence([torch.tensor(l_t) for l_t in label_tokens], batch_first=True, padding_value = self.IGNORE_INDEX)
        loss_tokens.update({"labels":labels})

        if self.context_enc:
            if self.context_prompt is not None:
                context_tokens = self.enc_tokenizer([self.context_prompt.format(**sample) for sample in batch], return_tensors = "pt", padding = True, truncation = True)
            else:
                context_tokens = self.enc_tokenizer([sample[self.context_column] for sample in batch], return_tensors = "pt", padding = True, truncation = True)
            gen_tokens = {"decoder_"+k if k!="labels" else k:v for k,v in gen_tokens.items()}
            loss_tokens = {"decoder_"+k if k!="labels" else k :v for k,v in loss_tokens.items()}
            gen_tokens.update(context_tokens)
            loss_tokens.update(context_tokens)
        
        answers = [b[self.answer_column] for b in batch]
        
        return loss_tokens, gen_tokens, answers
    
def extract_generated_text(batch, gen_out, tokenizer):
    """Extracts the only the newly generated text i.e. input text is removed"""
    batch_input_ids = batch["decoder_input_ids"] if "decoder_input_ids" in batch else batch["input_ids"]
    batch_input_ids = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
    gen_out = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
    gen_out = [seq[len(input_ids):] for seq, input_ids in zip(gen_out, batch_input_ids)]
    return gen_out

def evaluate(model, 
             tokenizer, 
             enc_tokenizer = None, 
             dataset_path = "squad",
             prompt_type = "q", 
             context_enc = False, 
             cover_labels = False,
             batch_size = 4, 
             context_column = "context",
             answer_column = "answers",
             run_decoder_only = False,
             max_batches = 50,
             save_logits = True, 
             num_prefix_token = 0):
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    full_batch_stats = []    
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    dataset = load_from_disk(dataset_path) if os.path.isdir(dataset_path) else load_dataset(dataset_path)
    dataset = dataset["validation"] if "validation" in dataset else dataset["test"]
    max_new_tokens = 60 if "cnn" not in dataset_path else 400

    bertscore = load("bertscore")
    bleu = load("bleu")
    rouge = load("rouge")

    if "squad" in dataset_path:
        dataset = dataset.map(lambda x: {"answers": x["answers"]["text"][0]})
    elif "cnn" in dataset_path:
        column_names = dataset.column_names
        if "article_init" in column_names: column_names.remove("article_init")
        dataset = dataset.map(lambda x: {"summary": [" ".join(entry.split()[:100]) for entry in x["highlights"]]}, batched=True)
        dataset = dataset.map(lambda x: {"article": [" ".join(entry.split()[:200]) for entry in x["article_half"]]}, batched=True, remove_columns=column_names)
    loader = DataLoader(dataset, batch_size = batch_size, collate_fn = EvalCollator(tokenizer,
                                                                                    enc_tokenizer, 
                                                                                    mode = prompt_type, 
                                                                                    context_enc = context_enc, 
                                                                                    cover_labels=cover_labels, 
                                                                                    context_column = context_column,
                                                                                    answer_column = answer_column,
                                                                                    num_prefix_token = num_prefix_token))

    for i,(loss_batch, gen_batch, answers) in tqdm(enumerate(loader)):
        loss_batch = {k:v.to(model.device) for k,v in loss_batch.items()}
        gen_batch = {k:v.to(model.device) for k,v in gen_batch.items()}
        out = model(**loss_batch)
        loss = out.loss.item()

        # Get the logits for only the answer tokens
        #len_answers = [len(t)-1 for t in tokenizer(answers).input_ids]
        #answer_logits = [logits[-l_answer:,:] for logits, l_answer in zip(out.logits, len_answers)]
        len_answers = [len(tokenizer(" "+a)["input_ids"])-1 for a in answers]
        attn_key = "decoder_attention_mask" if "decoder_attention_mask" in loss_batch else "attention_mask"
        answer_logits = [logit[attn_mask ==1][-answer_len:] for logit, attn_mask, answer_len in zip(out.logits, loss_batch[attn_key], len_answers)]


        # Adapter Attentions
        if hasattr(out, "adapter_attentions"):
            attn = out.adapter_attentions
            if len(attn) > 0:
                attn = torch.stack(attn)
        else:
            attn = []
            
        # Adapter Mean
        if hasattr(out, "adapter_mean"):
            adapter_mean = out.adapter_mean
            if len(adapter_mean) > 0:
                adapter_mean = torch.stack(adapter_mean)
        else:
            adapter_mean = []

        gen_out = model.generate(**gen_batch, max_new_tokens = max_new_tokens, eos_token_id= tokenizer.eos_token_id, output_scores  = True, return_dict_in_generate = True)
        # Equal to return_full_text = False -> Extract generated text
        gen_answers = extract_generated_text(gen_batch, gen_out.sequences, tokenizer)
        gen_logits = torch.stack(gen_out.scores)

        if run_decoder_only:
            decoder_loss_batch = {k.replace("decoder_",""):v for k,v in loss_batch.items() if "decoder" in k or "labels" in k}
            decoder_out = model.decoder(**decoder_loss_batch)
            decoder_loss = decoder_out.loss.item()
            decoder_answer_logits = [logit[attn_mask ==1][-answer_len:] for logit, attn_mask, answer_len in zip(decoder_out.logits, decoder_loss_batch["attention_mask"], len_answers)]

            decoder_gen_batch = {k.replace("decoder_",""):v for k,v in gen_batch.items() if "decoder" in k or "labels" in k}
            decoder_gen_out = model.decoder.generate(**decoder_gen_batch, max_new_tokens = max_new_tokens, eos_token_id= tokenizer.eos_token_id, output_scores  = True, return_dict_in_generate = True)
            decoder_gen_answers = extract_generated_text(decoder_gen_batch, decoder_gen_out.sequences, tokenizer)
            decoder_gen_logits = torch.stack(decoder_gen_out.scores)



        for j, (gen_answer, ref, answer_logit) in enumerate(zip(gen_answers, answers, answer_logits)):
            # Calc Metrics
            batch_stats =  {"gen_batch": {k:v[j].detach().cpu().numpy() for k,v in gen_batch.items()},
                            "loss_batch": {k:v[j].detach().cpu().numpy() for k,v in loss_batch.items()},
                            "loss": loss,
                            "generated": gen_answer,
                            "reference": ref,
                            "answer_logits": answer_logit.detach().cpu().numpy() if save_logits else None,
                            "gen_logits": gen_logits[:,j,:].detach().cpu().numpy() if save_logits else None,
                            "adapter_attn": attn[:,j,:,:].detach().cpu().numpy() if len(attn) > 0 else None,
                            "adapter_mean": adapter_mean[:,j].detach().cpu().numpy() if len(adapter_mean) > 0 else None,
                            "exact_match": exact([gen_answer], [ref]),
                            "f1": f1(gen_answer, ref),
                            "solution_present": solution_present([gen_answer], [ref]),
                            "bleu": bleu.compute(predictions=[gen_answer], references=[ref]) if len(gen_answer) > 0 else 0,
                            "rouge": rouge.compute(predictions=[gen_answer], references=[ref]),
                            "bert_score": bertscore.compute(predictions=[gen_answer], references=[ref], lang="en"),
                            "id": j,
                            "batch_id":i}
            if run_decoder_only:
                batch_stats.update({"decoder_loss": decoder_loss,
                                    "decoder_generated": decoder_gen_answers[j],
                                    "decoder_answer_logits": decoder_answer_logits[j].detach().cpu().numpy(),
                                    "decoder_gen_logits": decoder_gen_logits[:,j,:].detach().cpu().numpy(),
                                   })
            full_batch_stats.append(batch_stats)
        if i == max_batches:break

    return pd.DataFrame(full_batch_stats)
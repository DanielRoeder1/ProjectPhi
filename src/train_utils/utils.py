prompt_qca = """\
{context}
{question}

Answer: {answers}"""

prompt_qc  = """\
{context}
{question}

Answer:"""

prompt_qa = """\
{question}

Answer: {answers}"""

prompt_q = """\
{question}

Answer:"""

prompt_article = """\
{article}"""

prompt_article_gen = """\
{article_init}"""

prompt_article_summary = """\
Summary: {summary} Article: {article}"""

prompt_article_summary_gen = """\
Summary: {summary} Article:"""

prompt_qc_enc = """\
Question: {question} Context: {context}"""


def prepare_dataset(examples, 
                    prompt, 
                    tokenizer, 
                    enc_tokenizer = None,
                    create_labels = False, 
                    return_answers = False, 
                    apply_tokenization = True, 
                    context_enc =False, 
                    context_column = "context",
                    answer_column = "answers",
                    enc_prompt = None, 
                    num_prefix_token = 0):
    #input_text = [prompt.format(question = q, context = c, answers = a) for q, c, a in zip(examples["question"], examples["context"], examples["answers"])]
    num_entries = len(next(iter(examples.values())))
    records = [{k:v[i] for k,v in examples.items()} for i in range(num_entries)]
    input_text = [prompt.format(**r) for r in records]

    if num_prefix_token -1 > 0:
        enc_prompt = "".join([f"<PREFIX{i}>" for i in range(num_prefix_token-1)]) +  enc_prompt
    
    if apply_tokenization:
        input_ids = tokenizer(input_text, return_attention_mask=False)
        
        if create_labels:
            answer_ids = tokenizer([" "+ e for e in examples[answer_column]], return_attention_mask=False)["input_ids"]
            labels = input_ids["input_ids"].copy()
            labels = [(len(l)+1-(len(a)))*[-100] + l[-len(a)+1:] for l, a in zip(labels, answer_ids)]
            input_ids.update({"labels": labels})
        if return_answers:
            input_ids.update({"answer": examples[answer_column]})
        if context_enc:
            input_ids["decoder_input_ids"] = input_ids["input_ids"].copy()
            if enc_prompt is None:
                input_ids["input_ids"] = enc_tokenizer(examples[context_column], return_attention_mask=False, truncation = True)["input_ids"]
            else:
                input_ids["input_ids"] = enc_tokenizer([enc_prompt.format(**r) for r in records], return_attention_mask=False, truncation = True)["input_ids"]
        return input_ids
    else:
        return {"text": [tokenizer.eos_token + t for t in input_text], "answer": examples[answer_column]}

from torch.nn.utils.rnn import pad_sequence
import torch

class CustomCollator:
    def __init__(self, dec_tokenizer,enc_tokenizer = None):
        self.enc_pad_token_id = enc_tokenizer.pad_token_id if enc_tokenizer is not None else None
        self.dec_pad_token_id = dec_tokenizer.pad_token_id

        self.dec_tokenizer = dec_tokenizer
        self.enc_tokenizer = enc_tokenizer
        self.IGNORE_INDEX = -100

    def __call__(self, batch):
        input_ids, labels, attention_mask  = None, None, None
        # Extract and pad sequences for each column
        is_enc_dec_input = "decoder_input_ids" in batch[0]

        if "input_ids" in batch[0]:
            tokenizer = self.enc_tokenizer if is_enc_dec_input else self.dec_tokenizer
            data = tokenizer.pad({"input_ids": [item['input_ids'] for item in batch]}, return_tensors = "pt")
            #input_ids = pad_sequence([torch.tensor(item['input_ids']) for item in batch], batch_first=True, padding_value = self.enc_pad_token_id if is_enc_dec_input else self.dec_pad_token_id)
            if "labels" in batch[0]:
                labels = pad_sequence([torch.tensor(item['labels']) for item in batch], batch_first=True, padding_value = self.IGNORE_INDEX)
            else:
                labels = pad_sequence([torch.tensor(item["decoder_input_ids" if is_enc_dec_input else "input_ids"]) for item in batch], batch_first=True, padding_value = self.IGNORE_INDEX)
            #attention_mask = input_ids.ne(self.dec_pad_token_id)
            data["labels"] = labels

        #data = {"input_ids": input_ids, 
        #        "attention_mask" : attention_mask, 
        #        "labels": labels}
        
        if is_enc_dec_input:
            padded = self.dec_tokenizer.pad({"input_ids": [item['decoder_input_ids'] for item in batch]}, return_tensors = "pt")
            data.update({"decoder_"+k: v for k, v in padded.items()})
            #decoder_input_ids = pad_sequence([torch.tensor(item['decoder_input_ids']) for item in batch], batch_first=True, padding_value = self.dec_pad_token_id)
            #attention_mask = decoder_input_ids.ne(self.dec_pad_token_id)

            #data.update({"decoder_input_ids": decoder_input_ids, 
            #              "decoder_attention_mask" : attention_mask})
        return data
    

from sentence_splitter import SentenceSplitter
splitter = SentenceSplitter(language='en')

def extract_sentence(context, answer_index):
    end = 0
    for sentence in splitter.split(text=context):
        end += len(sentence)
        if answer_index <= end:
            return sentence
    
    if answer_index >= end:
        return sentence
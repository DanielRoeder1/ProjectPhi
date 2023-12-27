from typing import Dict 
import transformers

# Adjusted from: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py#L65
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    cls_update: bool = False,
):
    """Resize tokenizer and embedding.

    Resizes token embeds, and intializes new embeds with mean of vocab
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        if cls_update and tokenizer.cls_token_id is not None:
            # For the prefix encoder set CLS embed for new prefix token
            cls_embed = input_embeddings[tokenizer.cls_token_id]
            input_embeddings[-num_new_tokens:] = cls_embed
        else:
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            input_embeddings[-num_new_tokens:] = input_embeddings_avg
        # Encoder models may not have output embeds
        if (output_embeds:=model.get_output_embeddings()):
            output_embeddings = output_embeds.weight.data
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

import pandas as pd

def prepare_dataset(df_pd):
    unique_context = df_pd.context.unique()
    context_id_lookup = {context:i for i, context in enumerate(unique_context)}
    df_pd["context_id"] = df_pd.context.apply(lambda x: context_id_lookup[x])
    df_pd["answers"] = df_pd.apply(lambda x: x["answers"]["text"][0], axis = 1)
    context_answer_lookups = {context_id: context_group.answers.values for context_id, context_group in df_pd.groupby("context_id")}
    df_pd["false_answers"] = df_pd.apply(lambda x: [answer for answer in context_answer_lookups[x["context_id"]] if answer != x.answers], axis = 1)
    df_pd = df_pd.drop(["id", "title", "context_id"], axis = 1)
    return df_pd

def create_classification_records(df, context_column = "context"):
    df = prepare_dataset(df)
    all_records = []
    for i, row in df.iterrows():
        false_records = [{"context":row[context_column], "question": row["question"], "answers": false_answer, "label": 0} for false_answer in row["false_answers"][:1]]
        correct_record = {"context":row[context_column], "question": row["question"], "answers": row["answers"], "label": 1}
        all_records.extend(false_records)
        all_records.append(correct_record)

    df = pd.DataFrame(all_records)
    return df

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput

class CustomClassifier(nn.Module):
    def __init__(self, enc_model, num_prefix_tokens, num_labels=2):
        super().__init__()
        hidden_dim = enc_model.config.hidden_size
        self.enc_model = enc_model
        self.fusion_layer = nn.Linear(hidden_dim*2, hidden_dim)
        self.classif_layer = nn.Linear(hidden_dim*num_prefix_tokens, num_labels)
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_labels
        self.num_prefix_tokens = num_prefix_tokens
    
    def forward(self, qc_input_ids, qc_attention_mask, a_input_ids, a_attention_mask, labels= None):
        seq1_states = self.enc_model(input_ids = qc_input_ids, attention_mask = qc_attention_mask).last_hidden_state[:,:self.num_prefix_tokens,:]
        seq2_states = self.enc_model(input_ids = a_input_ids, attention_mask = a_attention_mask).last_hidden_state[:,:self.num_prefix_tokens,:]

        concat_seq = torch.cat((seq1_states, seq2_states), dim=-1)
        concat_seq = self.fusion_layer(concat_seq)
        concat_seq = nn.functional.relu(concat_seq)
        concat_seq = concat_seq.reshape(len(qc_input_ids), -1)
        concat_seq = self.dropout(concat_seq)
        logits = self.classif_layer(concat_seq)

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
class ClassifierCollator:
    def __init__(self, tokenizer, prompt_qc, prompt_a):
        self.tokenizer = tokenizer
        self.prompt_qc = prompt_qc
        self.prompt_a = prompt_a
        
    def __call__(self, batch):
        qc_tokens = self.tokenizer([self.prompt_qc.format(**sample) for sample in batch], padding=True, truncation=True, return_tensors="pt")
        a_tokens = self.tokenizer([self.prompt_a.format(**sample) for sample in batch], padding=True, truncation=True, return_tensors="pt")
        labels = torch.tensor([sample["label"] for sample in batch])
        return {"qc_input_ids": qc_tokens["input_ids"], 
                "qc_attention_mask": qc_tokens["attention_mask"], 
                "a_input_ids": a_tokens["input_ids"], 
                "a_attention_mask": a_tokens["attention_mask"],
                "labels": labels}

default_train = False
if default_train:
    #########################################################################
        
    from transformers import AutoModel, AutoTokenizer

    num_prefix_tokens = 3
    special_token_dict = {"additional_special_tokens": [f"<PREFIX{i}>" for i in range(num_prefix_tokens-1)]}

    enc_model = AutoModel.from_pretrained("roberta-base")
    enc_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    smart_tokenizer_and_embedding_resize(special_token_dict, enc_tokenizer, enc_model, cls_update=True)
    model = CustomClassifier(enc_model, num_prefix_tokens=num_prefix_tokens, num_labels=2)

    #########################################################################
    from datasets import Dataset, DatasetDict, load_from_disk

    dataset = load_from_disk("squad_with_answer_sentence")

    df_pd_train = dataset["train"].to_pandas()
    df_pd_validation = dataset["validation"].to_pandas()

    df_train = create_classification_records(df_pd_train)
    df_val = create_classification_records(df_pd_validation)

    df_hf = DatasetDict({"train": Dataset.from_pandas(df_train), "validation": Dataset.from_pandas(df_val)})

    prompt_qc = "".join([f"<PREFIX{i}>" for i in range(num_prefix_tokens-1)])+"Question: {question} Context: {context}"
    prompt_a = "".join([f"<PREFIX{i}>" for i in range(num_prefix_tokens-1)])+"Answer: {answers}"

    #########################################################################

    from transformers import TrainingArguments
    from train_utils.Trainer import CustomTrainer, UpdateOutputDirCallback
    import wandb

    wandb.login(key = "f190694cef6354f5205256582202a2b16502a236")


    training_args = TrainingArguments(per_device_train_batch_size= 2,
                                    gradient_accumulation_steps= 32,
                                    warmup_steps= 500,
                                    num_train_epochs= 2,
                                    learning_rate= 1e-4,
                                    fp16= False,
                                    logging_steps= 100,
                                    evaluation_strategy= "epoch",
                                    save_strategy= "epoch",
                                    output_dir= "/netscratch/roeder/classifier_train",
                                    optim= "adamw_torch",
                                    remove_unused_columns=False,)

    training_args.watch_wandb = True

    trainer = CustomTrainer(
        model=model,
        args= training_args,
        train_dataset=df_hf["train"],
        eval_dataset=df_hf["validation"],
        tokenizer=enc_tokenizer,
        data_collator=ClassifierCollator(enc_tokenizer, prompt_qc, prompt_a),
        callbacks=[UpdateOutputDirCallback()],
    )

    trainer.train()

else:
    from datasets import Dataset, DatasetDict, load_from_disk

    dataset = load_from_disk("squad_with_answer_sentence")

    df_pd_train = dataset["train"].to_pandas()
    df_pd_validation = dataset["validation"].to_pandas()

    df_pd_train = prepare_dataset(df_pd_train)
    df_pd_validation = prepare_dataset(df_pd_validation)

    df_pd_train = df_pd_train[df_pd_train.false_answers.apply(len)>0]
    df_pd_validation = df_pd_validation[df_pd_validation.false_answers.apply(len)>0]

    from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
    from torch.utils.data import Dataset, DataLoader
    class CustomDataset(Dataset):
        def __init__(self, dataset, prompt_a, prompt_qc):
            self.dataset = dataset
            self.prompt_a = prompt_a
            self.prompt_qc = prompt_qc
        
        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            row = self.dataset.iloc[idx]
            query_text = self.prompt_qc.format(question=row["question"], context=row["context"])
            pos_text = self.prompt_a.format(answers=row["answers"])
            neg_text = [self.prompt_a.format(answers=f_answer) for f_answer in row["false_answers"]]

            return InputExample(texts=[query_text, pos_text, neg_text[0]])
        
    num_prefix_tokens  = 3

    model = SentenceTransformer("roberta-base")
    model.tokenizer.model_max_length = 512

    word_embedding_model = model._first_module()

    tokens = [f"<PREFIX{i}>" for i in range(num_prefix_tokens-1)]
    word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    prompt_qc = "".join([f"<PREFIX{i}>" for i in range(num_prefix_tokens-1)])+"Question: {question} Context: {context}"
    prompt_a = "".join([f"<PREFIX{i}>" for i in range(num_prefix_tokens-1)])+"Answer: {answers}"

    df = CustomDataset(df_pd_train, prompt_a, prompt_qc)
    loader = DataLoader(df, batch_size=32, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Train the model
    model.fit(train_objectives=[(loader, train_loss)],
            epochs=1,
            warmup_steps=200,
            use_amp=True,
            checkpoint_path="/netscratch/roeder/classifier_train",
            checkpoint_save_steps=len(loader),
            optimizer_params = {'lr': 1e-4},
            )
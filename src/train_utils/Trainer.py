from transformers import TrainerCallback, Trainer
from typing import Optional
import wandb
import os
import shutil
from omegaconf import OmegaConf

class UpdateOutputDirCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model, **kwargs):
        if wandb.run is not None:
            run_name = wandb.run.name
            if hasattr(args, "_frozen"): args._frozen = False
            args.output_dir = os.path.join(args.output_dir, f"run_{run_name}")
            if hasattr(args, "_frozen"): args._frozen = True
            if "watch_wandb" in args.__dict__ and args.watch_wandb:
                wandb.watch(model, log='gradients', log_freq=max(100, args.logging_steps))



            if "script_paths" in args.__dict__:
                for script_path in args.script_paths:
                    dir_name = os.path.join(args.output_dir,"code",os.path.dirname(script_path))
                    os.makedirs(dir_name, exist_ok=True)
                    shutil.copy(script_path, os.path.join(dir_name, os.path.basename(script_path)))
            
                code_artifact = wandb.Artifact(name = "train_code",type="code")
                code_artifact.add_file("train.py")
                code_artifact.add_file("phi/modeling_phi.py")
                code_artifact.add_file("train_utils/eval.py")
                code_artifact.add_file("train_utils/utils.py")
                wandb.log_artifact(code_artifact)

# TODO: Doesent work asno access to trainer self
class AdditionalEvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if hasattr(self, "config_args") and self.config_args is not None:
            from eval import evaluate
            print("------ Running additional evaluation ------")
            eval_result = evaluate(self.model, 
                                   self.tokenizer, 
                                   self.enc_tokenizer, 
                                   **self.config_args.data_args, 
                                   context_enc= self.config_args.model_args.is_enc_dec)
            print(f"Avg Exact Match: {eval_result.exact_match.mean()}")
            print(f"Avg F1: {eval_result.f1.mean()}")
            eval_result.to_csv(os.path.join(args.output_dir, f"interm_eval_results_{state.global_step}.csv"))

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.config_args = kwargs.pop("config_args", None)
        self.enc_tokenizer = kwargs.pop("enc_tokenizer", None)
        super().__init__(*args, **kwargs)
        

    def save_model(self, output_dir: Optional[str] = None, _internal_call = False):
        """Aditionally saves our custom model config"""
        
        super().save_model(output_dir, _internal_call)

        if hasattr(self.model, "args") and self.is_world_process_zero():
            if wandb.run is not None: self.model.args.wandb_run = wandb.run.name
            config_path = os.path.join(output_dir, "model_config.yaml")
            OmegaConf.save(self.model.args, config_path)

        if self.enc_tokenizer is not None and self.is_world_process_zero():
            enc_tokenizer_save_path = os.path.join(output_dir, "enc_tokenizer")
            self.enc_tokenizer.save_pretrained(enc_tokenizer_save_path)
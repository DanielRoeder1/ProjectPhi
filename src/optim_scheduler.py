from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import AdamW
import torch
import math

class DifferentialAlignmentSchedulerWithZeroPeriodLRFixed(_LRScheduler):
    def __init__(self, optimizer, warmup_steps_g1, warmup_steps_g2, lr_g1, total_steps, zero_period_steps, zero_period_lr, last_step=-1, decay = "cosine"):
        self.warmup_steps_g1 = warmup_steps_g1
        self.warmup_steps_g2 = warmup_steps_g2
        self.lr_g1 = lr_g1
        self.total_steps = total_steps
        self.zero_period = zero_period_steps
        self.zero_period_lr = zero_period_lr
        self.decay = decay
        super(DifferentialAlignmentSchedulerWithZeroPeriodLRFixed, self).__init__(optimizer, last_step)

    def get_lr(self):
        def cosine_annealing(step, total_steps, initial_lr):
            return 0.5 * initial_lr * (1 + math.cos(step / total_steps * math.pi))
        def linear_decay_schedule(current_step, total_steps, initial_lr):
            return initial_lr * (1 - current_step / total_steps)


        # Group 1 learning rate logic
        if self.last_epoch < self.warmup_steps_g1:
            scale_g1 = self.last_epoch / self.warmup_steps_g1
            lr_g1 = self.lr_g1 * scale_g1
        else:
            steps_since_warmup = self.last_epoch - self.warmup_steps_g1
            if self.decay == "linear":
                lr_g1 = linear_decay_schedule(steps_since_warmup, self.total_steps - self.warmup_steps_g1, self.lr_g1)
            elif self.decay == "cosine":
                lr_g1 = cosine_annealing(steps_since_warmup, self.total_steps - self.warmup_steps_g1, self.lr_g1)

        # Group 2 learning rate logic
        if self.last_epoch < self.zero_period:
            lr_g2 = self.zero_period_lr
        elif self.zero_period <= self.last_epoch < self.warmup_steps_g2 + self.zero_period:
            # Calculate the target LR that g2 should achieve by the end of its warmup
            target_lr_at_g2_end = lr_g1 if self.last_epoch >= self.warmup_steps_g1 else self.lr_g1
            scale_g2 = (self.last_epoch - self.zero_period) / self.warmup_steps_g2
            lr_g2 = self.zero_period_lr + (target_lr_at_g2_end - self.zero_period_lr) * scale_g2
        else:
            lr_g2 = lr_g1  # Align with Group 1

        # Return each group twice to account for decay / no decay param groups
        return (lr_g1, lr_g1, lr_g2, lr_g2)

def get_optim_scheduler(model, new_params, optim_args, total_steps):

    weight_decay = optim_args.weight_decay
    dummy_lr = 1e-9 # Is overwritten by the CustomScheduler

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n.replace("dec_model.","") in new_params["missing_keys"] and p.requires_grad and not any([x in n for x in ["bias", "ln"]]))
            ],
            "weight_decay": weight_decay,
            'lr': dummy_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n.replace("dec_model.","") in new_params["missing_keys"] and p.requires_grad and any([x in n for x in ["bias", "ln"]]))
            ],
            "weight_decay": 0.0,
            'lr': dummy_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n.replace("dec_model.","") not in new_params["missing_keys"] and p.requires_grad and not any([x in n for x in ["bias", "ln"]]))
            ],
            "weight_decay": weight_decay,
            'lr': dummy_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n.replace("dec_model.","") not in new_params["missing_keys"] and p.requires_grad and any([x in n for x in ["bias", "ln"]]))
            ],
            "weight_decay": 0.0,
            'lr': dummy_lr
        },
                ]

    optimizer = AdamW(optimizer_grouped_parameters)

    scheduler = DifferentialAlignmentSchedulerWithZeroPeriodLRFixed(optimizer, 
                                                                    warmup_steps_g1= optim_args.warmup_steps_g1, 
                                                                    warmup_steps_g2= optim_args.warmup_steps_g2, 
                                                                    lr_g1= optim_args.lr_g1,
                                                                    total_steps=total_steps, 
                                                                    zero_period_steps = optim_args.zero_period_steps, 
                                                                    zero_period_lr = optim_args.zero_period_lr,
                                                                    decay = optim_args.decay)
    return optimizer, scheduler

if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from types import SimpleNamespace

    optim_args = SimpleNamespace(weight_decay = 0.01, 
                                warmup_steps_g1= 500, 
                                warmup_steps_g2= 2000, 
                                zero_period_steps= 4000, 
                                total_steps= 12_800, 
                                lr_g1= 1e-4, 
                                zero_period_lr= 0,
                                decay = "cosine")
    import sys
    sys.path.append("..")
    from modeling_gpt2 import GPT2LMHeadModel
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("gpt2")
    config.know_layer = [1,2,3,4]
    config.know_type = "crossattn"
    config.hidden_dropout = 0.1

    model, new_params = GPT2LMHeadModel.from_pretrained("gpt2",  config = config, output_loading_info = True)
    optimizer, scheduler = get_optim_scheduler(model, new_params, optim_args, optim_args.total_steps)

    # Initialize learning rates for both groups
    lr_g1 = []
    lr_g2 = []

    # Simulate the training loop and collect learning rates
    for step in range(optim_args.total_steps):  # Example total steps
        lr_g1.append(scheduler.get_lr()[0])
        lr_g2.append(scheduler.get_lr()[2])
        scheduler.step()

    plt.figure(figsize=(10, 6))
    plt.plot(lr_g1, label="Group 1 (g1)", color='blue')
    plt.plot(lr_g2, label="Group 2 (g2)", color='red')

    # Increase font size for labels and title
    plt.xlabel("Steps", fontsize=14)  # Change fontsize as needed
    plt.ylabel("Learning Rate", fontsize=14)  # Change fontsize as needed
    plt.title("Dual Group Learning Rate Scheduling", fontsize=16)  # Change fontsize as needed

    # Increase font size for legend
    plt.legend(fontsize=12)  # Change fontsize as needed

    plt.grid(True)
    plt.show()
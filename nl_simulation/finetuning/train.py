

from datasets import load_from_disk
from transformers.training_args import TrainingArguments
from trl import SFTConfig, SFTTrainer,DataCollatorForCompletionOnlyLM
from peft import LoraConfig,get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer,TrainerCallback
from accelerate import Accelerator
from simple_parsing import ArgumentParser


args = ArgumentParser()
args.add_argument("--dataset", type=str, default="quantiles_top5")
args = args.parse_args()

accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(project_name="finetune")
dataset = load_from_disk(f"simulation_data_{args.dataset}")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",load_in_8bit=True)

tokenizer.pad_token = "<|finetune_right_pad_id|>"
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['samples'])):
        text = f"{example['samples'][i]}{example['labels'][i]}<|eot_id|>"
        output_texts.append(text)
    return output_texts

class AccuracyMetric():
    def __init__(self):
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def update(self,logits,labels):
        #logits = logits.cpu().numpy()
        #labels = labels.cpu().numpy()
        # Mask out -100 values
        
        for i in range(len(logits)):
            if len(logits[i][labels[i] != -100]) == 0:
                continue
            prediction = logits[i][labels[i] != -100][-3].argmax().item() # -3 is the index of the response, -2 is the index of <eot_id>
            label = labels[i][labels[i] != -100][-2].item() # -1 is the index of <eot_id>
            
            if prediction == label:
                if label == 15: # this is the token index for "1"
                    self.true_positives += 1
                else:
                    self.true_negatives += 1
            else:
                if label == 15:
                    self.false_positives += 1
                else:
                    self.false_negatives += 1
        
    def compute(self):
        if self.true_positives+self.false_negatives == 0:
            recall = 0
        else:
            recall = self.true_positives/(self.true_positives+self.false_negatives)
        if self.true_positives+self.false_positives == 0:
            precision = 0
        else:
            precision = self.true_positives/(self.true_positives+self.false_positives)
        if self.true_negatives+self.false_positives == 0:
            balanced_accuracy = 0
        else:
            balanced_accuracy = (recall+self.true_negatives/(self.true_negatives+self.false_positives))/2
        if recall+precision == 0:
            f1_score = 0
        else:
            f1_score = 2*recall*precision/(recall+precision)
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        return {"balanced_accuracy": balanced_accuracy, "f1_score": f1_score,"precision": precision,"recall": recall}
def compute_metrics(eval_pred,compute_result):
    acc_metric.update(eval_pred.predictions,eval_pred.label_ids)
    if compute_result:
        return acc_metric.compute()




acc_metric = AccuracyMetric()
response_template = "<|start_header_id|>assistant<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

peft_config = LoraConfig(
    
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM",
    lora_dropout=0.1
)
peft_model = get_peft_model(model,peft_config)


#shuffle dataset
dataset["train"] = dataset["train"].shuffle(seed=42)
dataset["test"] = dataset["test"].shuffle(seed=42).select(range(5000))
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    args=SFTConfig(output_dir=f"./output_{args.dataset}",
                   #auto_find_batch_size=True
                   per_device_eval_batch_size=6,
                   per_device_train_batch_size=4,
                   gradient_accumulation_steps=4
                   ,eval_strategy="steps",eval_steps=100,eval_on_start=False,logging_steps=50,max_steps=1000,max_seq_length=512,batch_eval_metrics=True,save_strategy="steps",save_steps=100),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    compute_metrics=compute_metrics
)
trainer.train()

accelerator.end_training()
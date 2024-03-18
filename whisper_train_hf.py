"""
Test codes for fine-tuning Whisper speech-to-text, i.e. speech recognition 
"""

from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from evaluate import load


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
        
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
    
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    # print(f"""** {batch}""")
    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    # print(f"""** {batch["labels"]}""")
    return batch

#Step-1: Define model structure & initialization, feature extractor, text tokenizer
model_base_default = "openai/whisper-small"
language = "zh"
save_dir = "/home/gs/work/audiolm-pytorch/whisper-small-zh-me"
max_steps = 500

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_base_default, language=language)
tokenizer = WhisperTokenizer.from_pretrained(model_base_default, task="transcribe", language=language)
processor = WhisperProcessor.from_pretrained(model_base_default, task="transcribe", language=language)

#Step-2: Prepare train/dev/test data
common_voice1 = DatasetDict()
common_voice1["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "zh-CN", split="train", use_auth_token=False).select(range(1000))
common_voice1["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "zh-CN", split="test", use_auth_token=False).select(range(50))
common_voice = common_voice1.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")                                             

#
model = WhisperForConditionalGeneration.from_pretrained(model_base_default)
model.generation_config.language = language

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


training_args = Seq2SeqTrainingArguments(
    output_dir=save_dir, #"./whisper-small-zh-me",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=100, #500,
    max_steps=max_steps, #4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500, #500, #1000,
    eval_steps=500, #500, #1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False, #True,


)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
trainer.train()

#evaluate the model trained
#evaluation

# from datasets import load_dataset
# from transformers import WhisperForConditionalGeneration, WhisperProcessor
# import torch
# from evaluate import load
# from datasets import Audio

# librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test")

model_pth = f"{save_dir}/checkpoint-{max_steps}" #(
model_token_pth = "openai/whisper-small"
# model_pth = "openai/whisper-large"
is_local = True
processor = WhisperProcessor.from_pretrained(model_token_pth)#"openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained(model_pth, local_files_only=is_local).to("cuda")

def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    batch["reference"] = processor.tokenizer._normalize(batch['sentence'])

    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"))[0]
    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer._normalize(transcription)
    return batch

common_voice2 = common_voice1["test"].cast_column("audio", Audio(sampling_rate=16000))
result = common_voice2.map(map_to_pred)

wer = load("wer")
print(100 * wer.compute(references=result["reference"], predictions=result["prediction"]))

                                             #
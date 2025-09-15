# LLM-JEPA

## Set Up

See `setup.sh`.

**NOTE**: Do NOT run `setup.sh` directly. Read the file, choose the configurtion for your envirnoment, and execute the relevant commands manually.

## LLM-JEPA Fine-tuning

The fine-tuning script is in `finetune.py`. A convenient driver script, `run.sh`, provides `run_regular()` for standard fine-tuning, and `run_jepa()` for LLM-JEPA fine-tuning.

For all experiments, we fix number of epochs to 4. The `last_token` setting depends on the model family; see the commented lines in `run.sh` for how to set it. Each configuration is run with 5 random seeds. We report mean accuracy and standard deviation.

The original implementation required two additional forward passes to encode `Text` and `Code` separately. The latest version combines them into a single forward pass using a 4D additive attention mask. Enable this feature with `--additive_mask`.

## Large models

Similarly, we provide `finetune8bh200.py` and `run8bh200.sh` for training modesl up to 8B parameters on NVIDIA H200 GPUs.

## LLM-JEPA with LoRA

Use `--lora` and `--lora_rank <N>` to enable LoRA fine-tuning for LLM-JEPA.

## Pretraining

Use `--pretrain` to start from randomly initialized weights.

For pretraining on the `paraphrase` dataset, pass `--plain --trainall` to disable the OpenAI message format, train next-token prediction, and jointly minimize distances between paraphrase variants.

After pretaining, fine-tune with `--plain` on `rotten_tomatoes` and `yelp`. For evaluation, run with `--plain --statswith` to bypass the OpenAI message format and score only the first token(the model isn't instruction-tuned, so it may not emit a clean stop). 

## Datasets

Most datasets include `_train.jsonl` and `_test.jsonl` files for fine-tuning and evaluation, repsectively. The originals come from prior publications; we preprocessed them and include the results here for convenience.

*  `synth` and `turk`, from https://arxiv.org/abs/1608.03000
*  `gsm8k`, from https://arxiv.org/abs/2110.14168
*  `spider`, from https://arxiv.org/abs/1809.08887. You aslo need to unzip `spider_data.zip` which contains `sqlite` databases to execute and evaluate the generated queries.
*  `paraphrase`, from HuggingFace `cestwc/paraphrase` dataset. Only have `train` split, for pre-training only.
*  `rotten_tomatoes`, from HuggingFace `cornell-movie-review-data/rotten_tomatoes` dataset. Used for fine-tuning and evaluating models pretrained by `paraphrase` dataset.
*  `yelp`, from HuggingFace `Yelp/yelp_review_full` dataset. Used for fine-tuning and evaluating models pretrained by `paraphrase` dataset.

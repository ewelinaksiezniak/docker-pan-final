
Submit to TIRA via:

```
tira-cli code-submission \
	--mount-hf-model Ewel/model_trained_on_contrastive_encoder_10_epoch_question_medium_freeze_2  Ewel/modeltrained_on_contrastive_encoder_10_epoch_quote_easy_freeze_0 FacebookAI/xlm-roberta-base Ewel/model_trained_on_contrastive_encoder_10_epoch_question_freeze_0 \
	--path . --task multi-author-writing-style-analysis-2025 --dataset multi-author-writing-spot-check-20250503-training \
	--command 'python3 MySoft.py -i $inputDataset -o $outputDir' --dry-run
```


Submit to TIRA via:

```
tira-cli code-submission \
	--mount-hf-model FacebookAI/xlm-roberta-base \
	--path . --task multi-author-writing-style-analysis-2025 --dataset multi-author-writing-spot-check-20250503-training \
	--command 'python3 MySoft.py -i $inputDataset -o $outputDir' --dry-run
```

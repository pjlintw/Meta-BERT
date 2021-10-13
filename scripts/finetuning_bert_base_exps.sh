#!/bin/sh
# Run fine-tuning script`

for num_domain in 1 3 5 50;
do
	echo "Execute fine-tuning script with ${num_domain} domains."
	python run_finetuning.py \
		--output_dir results/transfer.init-bert.nd-$num_domain \
		--epochs 10 \
		--num_domain $num_domain \
		--gpu_id 1
done





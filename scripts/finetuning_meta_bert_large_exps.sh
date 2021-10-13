#!/bin/sh
# Run fine-tuning script on meta-leart BERT

declare -a arr=("100 1 results/meta.nt-100.nd-1/ckpt/meta.epoch-3.step-5.pt"
			    "100 3 results/meta.nt-100.nd-3/ckpt/meta.epoch-2.step-17.pt"
			    "100 5 results/meta.nt-100.nd-5/ckpt/meta.epoch-3.step-7.pt"
	 		    "100 50 results/meta.nt-100.nd-50/ckpt/meta.epoch-1.step-1.pt"
			    )


for ele in "${arr[@]}";
do
	read -a strarr <<< "$ele";
	for num_domain in 1 3 5 50;
	do
		echo "Execute fine-tuning script with ${num_domain} domains"
		echo "Model is initialized from checkpoint ${pt_file}"

		python run_finetuning.py \
			--bert_model ${strarr[2]} \
			--output_dir results/transfer.init-meta-t${strarr[0]}-d${strarr[1]}.nd-$num_domain \
			--epochs 10 \
			--num_domain ${num_domain} \
			--gpu_id 1
	done
done



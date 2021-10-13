#!/bin/sh
# Run fine-tuning script on meta-leart BERT

declare -a arr=("10 1 results/meta.nt-10.nd-1/ckpt/meta.epoch-3.step-1.pt"
				"10 3 results/meta.nt-10.nd-3/ckpt/meta.epoch-2.step-1.pt"
	 			"10 5 results/meta.nt-10.nd-5/ckpt/meta.epoch-2.step-1.pt"
	 			"10 50 results/meta.nt-10.nd-50/ckpt/meta.epoch-1.step-1.pt"

				"50 1 results/meta.nt-50.nd-1/ckpt/meta.epoch-1.step-7.pt"
			 	"50 3 results/meta.nt-50.nd-3/ckpt/meta.epoch-3.step-3.pt"
			 	"50 5 results/meta.nt-50.nd-5/ckpt/meta.epoch-2.step-5.pt"
	 			"50 50 results/meta.nt-50.nd-50/ckpt/meta.epoch-3.step-9.pt"
				
				"100 1 results/meta.nt-100.nd-1/ckpt/meta.epoch-3.step-5.pt"
			    "100 3 results/meta.nt-100.nd-3/ckpt/meta.epoch-2.step-17.pt"
			    "100 5 results/meta.nt-100.nd-5/ckpt/meta.epoch-3.step-7.pt"
	 		    "100 50 results/meta.nt-100.nd-50/ckpt/meta.epoch-1.step-1.pt"
				)


for ele in "${arr[@]}";
do
	read -a strarr <<< "$ele";
	echo "Execute zero-shot script"
	echo "Model is initialized from checkpoint ${strarr[2]}"

	python run_finetuning.py \
		--bert_model ${strarr[2]} \
		--output_dir results/transfer.init-meta-t${strarr[0]}-d${strarr[1]}.zero-shot \
		--zero_shot True

done



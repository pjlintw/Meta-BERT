#!/bin/sh
# Run meta-training script`

for num_domain in 1 3 5 50;
do
	for num_task in 100;
	do
		echo "Execute meta-training script with ${num_task} training tasks from ${num_domain} domains."
		python run_meta_training.py \
			--output_dir results/meta.nt-$num_task.nd-$num_domain \
			--num_train_task $num_task \
			--num_domain $num_domain \
			--gpu_id 2
	done
done



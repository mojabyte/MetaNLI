# MetaNLI

Meta learning algorithms to train cross-lingual NLI (multi-task) models

## Train (source task)

### Reptile

To train the model using Reptile algorithm, run the command below:

```bash
python reptile.py \
    --meta_tasks sc_en,sc_de,sc_es,sc_fr \
    --queue_len 4 \
    --temp 5.0 \
    --epochs 1 \
    --meta_lr 1e-5 \
    --scheduler \
    --gamma 0.5 \
    --step_size 4000 \
    --shot 4 \
    --meta_iteration 8000 \
    --log_interval 300
```

### Prototypical

To train the model using Prototypical Networks algorithm, run the command below:

```bash
python prototype.py \
    --meta_tasks sc_en,sc_de,sc_es,sc_fr \
    --target_task sc_fa \
    --epochs 1 \
    --meta_lr 1e-5 \
    --lambda_1 1 \
    --lambda_2 1 \
    --scheduler \
    --gamma 0.5 \
    --step_size 1000 \
    --shot 8 \
    --query_num 0 \
    --target_shot 8 \
    --meta_iteration 2500 \
    --log_interval 50
```

## Zero-shot Test (on target task)

To perform a zero-shot test of the trained model on the target task, run the command below:

```bash
python zeroshot.py \
    --load saved/model_sc.pt \
    --task sc_fa
```

## Fine-tune (target task)

To fine-tune the trained model on the target task, run the command below:

```bash
python finetune.py \
    --save saved \
    --model_filename fine.pt \
    --load saved/model_sc.pt \
    --task sc_fa \
    --epochs 5 \
    --lr 1e-5
```

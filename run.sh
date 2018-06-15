# pretrain
CUDA_VISIBLE_DEVICES=0 python main.py --name=pretrain_b32_gf32_lr4_10k_3 --is_train=True --learning_rate=0.0001  --batch_size=32 --gf_dim=32 --is_sup_train=True --max_iter=1000

# finetune
#CUDA_VISIBLE_DEVICES=0 python main.py --name=finetune_lr6 --learning_rate=0.000001 --batch_size=1 --model_dir=checkpoint/pretrain_b32_gf32_lr4_10k --gf_dim=32 --is_sup_train=False --is_dryrun=False --is_train=True --key_loss=True --max_iter=1000 #--silh_loss=True

# Predict 1
CUDA_VISIBLE_DEVICES=0 python main.py --name=pretrain_b32_gf32_lr4_10k_3 --model_dir=checkpoint/pretrain_b32_gf32_lr4_10k_3 --is_train=False  --batch_size=1 --gf_dim=32 --is_sup_train=False

# Predict 2
#CUDA_VISIBLE_DEVICES=0 python main.py --name=finetune_lr6 --batch_size=1 --model_dir=checkpoint/pretrain_b32_gf32_lr4_10k --gf_dim=32 --is_sup_train=False --is_train=False #--key_loss=True #--silh_loss=True

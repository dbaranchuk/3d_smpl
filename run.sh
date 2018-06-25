# pretrain
#CUDA_VISIBLE_DEVICES=0 python main.py --name=model0 --model_dir=checkpoint/pretrain_b32_gf32_lr4_10k --is_train=True --learning_rate=0.000001  --batch_size=128 --gf_dim=32 --is_sup_train=True --max_iter=2000

CUDA_VISIBLE_DEVICES=0 python main.py --name=model8 --model_dir=checkpoint/pretrain_b32_gf32_lr4_10k --is_train=True --learning_rate=0.00001  --batch_size=64 --gf_dim=32 --is_sup_train=True --max_iter=2000

CUDA_VISIBLE_DEVICES=0 python main.py --name=model8 --model_dir=checkpoint/model8 --is_train=True --learning_rate=0.000001  --batch_size=64 --gf_dim=32 --is_sup_train=True --max_iter=2000

CUDA_VISIBLE_DEVICES=0 python main.py --name=model8 --model_dir=checkpoint/model8 --batch_size=1 --gf_dim=32 --is_sup_train=False --is_train=False

#finetune
rm -rf checkpoint/finetune1/
CUDA_VISIBLE_DEVICES=0 python main.py --name=finetune1 --batch_size=1 --model_dir=checkpoint/model8 --gf_dim=32 --is_sup_train=False --is_train=True --learning_rate=0.00001 --key_loss=True --max_iter=1000  #--silh_loss=True

CUDA_VISIBLE_DEVICES=0 python main.py --name=finetune1 --batch_size=1 --model_dir=checkpoint/finetune1 --gf_dim=32 --is_sup_train=False --is_train=False

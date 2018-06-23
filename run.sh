# pretrain
#CUDA_VISIBLE_DEVICES=0 python main.py --name=pretrain_b64_gf16_lr4_10k --learning_rate=0.0001  --batch_size=64 --is_train=True --gf_dim=16 --is_sup_train=True --max_iter=10000

#CUDA_VISIBLE_DEVICES=0 python main.py --name=model0 --model_dir=checkpoint/pretrain_b32_gf32_lr4_10k --is_train=True --learning_rate=0.000001  --batch_size=128 --gf_dim=32 --is_sup_train=True --max_iter=2000

#CUDA_VISIBLE_DEVICES=0 python main.py --name=model4 --model_dir=checkpoint/pretrain_b32_gf32_lr4_10k --is_train=True --learning_rate=0.00001  --batch_size=64 --gf_dim=32 --is_sup_train=True --max_iter=2000

#CUDA_VISIBLE_DEVICES=0 python main.py --name=model4 --model_dir=checkpoint/model4 --is_train=True --learning_rate=0.000001  --batch_size=64 --gf_dim=32 --is_sup_train=True --max_iter=2000

#finetune
CUDA_VISIBLE_DEVICES=0 python main.py --name=finetuned --batch_size=1 --model_dir=checkpoint/model4 --gf_dim=32 --is_sup_train=False --is_train=True --key_loss=True --silh_loss=True --max_iter=1000

#CUDA_VISIBLE_DEVICES=0 python main.py --name=finetuned --batch_size=1 --model_dir=checkpoint/finetuned --gf_dim=32 --is_sup_train=False --is_train=False 

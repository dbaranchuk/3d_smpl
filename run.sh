# pretrain
#CUDA_VISIBLE_DEVICES=0 python main.py --name=model0 --model_dir=checkpoint/pretrain_b32_gf32_lr4_10k --is_train=True --learning_rate=0.000001  --batch_size=128 --gf_dim=32 --is_sup_train=True --max_iter=2000

#CUDA_VISIBLE_DEVICES=0 python main.py --name=model7 --model_dir=checkpoint/pretrain_b32_gf32_lr4_10k --is_train=True --learning_rate=0.00001  --batch_size=64 --gf_dim=32 --is_sup_train=True --max_iter=1000

#CUDA_VISIBLE_DEVICES=0 python main.py --name=model7 --model_dir=checkpoint/model7 --is_train=True --learning_rate=0.000001  --batch_size=64 --gf_dim=32 --is_sup_train=True --max_iter=1000

#CUDA_VISIBLE_DEVICES=0 python main.py --name=model7 --model_dir=checkpoint/model7 --batch_size=1 --gf_dim=32 --is_sup_train=False --is_train=False

#finetune
#rm -rf checkpoint/finetune/
#CUDA_VISIBLE_DEVICES=0 python main.py --name=finetune --batch_size=1 --model_dir=checkpoint/model7 --gf_dim=32 --is_sup_train=False --is_train=True --learning_rate=0.0000000000000001 --silh_loss=True --max_iter=1000  #--key_loss=True

CUDA_VISIBLE_DEVICES=0 python main.py --name=finetune --batch_size=1 --model_dir=checkpoint/finetune --gf_dim=32 --is_sup_train=False --is_train=False

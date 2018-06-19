# pretrain
#CUDA_VISIBLE_DEVICES=0 python main.py --name=model0 --model_dir=checkpoint/pretrain_b32_gf32_lr4_10k --is_train=True --learning_rate=0.000001  --batch_size=128 --gf_dim=32 --is_sup_train=True --max_iter=2000

#CUDA_VISIBLE_DEVICES=0 python main.py --name=overfit --model_dir=checkpoint/model0 --is_train=True --learning_rate=0.000001  --batch_size=32 --gf_dim=32 --is_sup_train=True --max_iter=1000

CUDA_VISIBLE_DEVICES=0 python main.py --name=overfit --batch_size=1 --model_dir=checkpoint/overfit --gf_dim=32 --is_sup_train=False --is_train=False

#CUDA_VISIBLE_DEVICES=0 python main.py --name=model0 --batch_size=1 --model_dir=checkpoint/model0 --gf_dim=32 --is_sup_train=False --is_train=False #--key_loss=True #--silh_loss=True

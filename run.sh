# pretrain
CUDA_VISIBLE_DEVICES=0 python main.py --name=model0 --model_dir=checkpoint/pretrain_b32_gf32_lr4_10k --is_train=True --learning_rate=0.000001  --batch_size=32 --gf_dim=32 --is_sup_train=True --max_iter=5000


#CUDA_VISIBLE_DEVICES=0 python main.py --name=model0 --batch_size=1 --model_dir=checkpoint/model2 --gf_dim=32 --is_sup_train=False --is_train=False #--key_loss=True #--silh_loss=True

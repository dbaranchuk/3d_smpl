# pretrain
CUDA_VISIBLE_DEVICES=0 nohup python main.py --name=pretrain_b64_gf16_lr4_10k --learning_rate=0.0001  --batch_size=64 --is_train=True --gf_dim=16 --is_sup_train=True --max_iter=10000

#CUDA_VISIBLE_DEVICES=0 python main.py --name=model0 --model_dir=checkpoint/pretrain_b32_gf32_lr4_10k --is_train=True --learning_rate=0.000001  --batch_size=128 --gf_dim=32 --is_sup_train=True --max_iter=2000

#CUDA_VISIBLE_DEVICES=0 python main.py --name=model3 --model_dir=checkpoint/pretrain_b32_gf32_lr4_10k --is_train=True --learning_rate=0.00001  --batch_size=64 --gf_dim=32 --is_sup_train=True --max_iter=4000

#CUDA_VISIBLE_DEVICES=0 python main.py --name=model2 --model_dir=checkpoint/pretrain_b32_gf32_lr4_10k --is_train=True --learning_rate=0.0000005  --batch_size=128 --gf_dim=32 --is_sup_train=True --max_iter=2000

#CUDA_VISIBLE_DEVICES=0 python main.py --name=overfit --model_dir=checkpoint/model0 --is_train=True --learning_rate=0.000001  --batch_size=32 --gf_dim=32 --is_sup_train=True --max_iter=2000

#CUDA_VISIBLE_DEVICES=0 python main.py --name=model2 --batch_size=1 --model_dir=checkpoint/model2 --gf_dim=32 --is_sup_train=False --is_train=False

#CUDA_VISIBLE_DEVICES=0 python main.py --name=model0 --batch_size=1 --model_dir=checkpoint/model0 --gf_dim=32 --is_sup_train=False --is_train=False #--key_loss=True #--silh_loss=True

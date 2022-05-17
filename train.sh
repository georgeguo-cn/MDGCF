python MDGCF.py --dataset=AMusic --gpu_id=0 --batch_size=2048 --layer=4 --top_H=4 --alpha=0.2 --beta=0.1 --self_loop=0
python MDGCF.py --dataset=AKindle --gpu_id=0 --batch_size=2048 --layer=4 --top_H=4 --alpha=0.3 --beta=0.3 --self_loop=0
python MDGCF.py --dataset=Gowalla --gpu_id=0 --batch_size=2048 --layer=3 --top_H=32 --alpha=0.2 --beta=0.3 --self_loop=0
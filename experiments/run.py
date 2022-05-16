import os

def run(dataset, network, num_layers, model_size, lr, beta1, beta2, eps, embedding_size,
        dim_feedforward, dropout, epochs, batch_size):
  os.system(f'''
    python3 ./main.py --dataset {dataset} --network {network} --num_layers {num_layers} --model_size {model_size} \
                     --lr {lr} --beta1 {beta1} --beta2 {beta2} --eps {eps} \
                     --embedding_size {embedding_size} \
                     --feedforward_dim {dim_feedforward} --dropout {dropout} \
                     --epochs {epochs} --batch_size {batch_size}
    ''')

num_layers = 1
network = 'gru'
model_size = 75
lr = 0.00001  # 0.0001 for Chinese
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
embedding_size = 100
dropout = 0.2
epochs = 150
dim_feedforward = 100
batch_size = 1

# TODO: fix
for dataset in ['romance_orthographic']:  # ['chinese', 'romance_orthographic', 'romance_phonetic']:
  run(dataset, network, num_layers, model_size, lr, beta1, beta2, eps, embedding_size,
      dim_feedforward, dropout, epochs, batch_size)
  # python main.py --dataset $dataset --network gru --num_layers $num_layers --model_size $hidden_dim --lr $lr --beta1 $beta1 --beta2 $beta2 --eps $eps --embedding_size $embedding_size --feedforward_dim $dim_feedforward --dropout $dropout --epochs $epochs --batch_size $batch_size

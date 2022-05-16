from main import *
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True,
                    help='chinese/romance_orthographic/romance_phonetic/austronesian')
args = parser.parse_args()
DATASET = args.dataset
MODELPATH_LOSS = f'./checkpoints/{DATASET}_best_loss.pt'
MODELPATH_ED = f'./checkpoints/{DATASET}_best_ed.pt'

# evaluate on the model with the best loss and the one with the best edit distance
for filepath, criterion in [(MODELPATH_LOSS, 'loss'), (MODELPATH_ED, 'edit distance')]:
    saved_info = load_model(filepath)
    args = saved_info['args']
    vocab = saved_info['vocab']

    # TODO: have all parameters been loaded?
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = args.epochs
    NUM_LAYERS = args.num_layers
    NETWORK = args.network
    LEARNING_RATE = args.lr
    BETA_1 = args.beta1
    BETA_2 = args.beta2
    EPS = args.eps
    EMBEDDING_SIZE = args.embedding_size
    DROPOUT = args.dropout
    MAX_LENGTH = 30 if 'romance' in DATASET else 15
    HIDDEN_SIZE = args.model_size
    FEEDFORWARD_DIM = args.feedforward_dim

    dev_dataset, _, _ = DataHandler.load_dataset(f'./data/{DATASET}/dev.pickle')
    test_dataset, _, langs = DataHandler.load_dataset(f'./data/{DATASET}/test.pickle')
    langs = langs + ['sep']
    L2I = {l: idx for idx, l in enumerate(langs)}
    dev_tensors = DataHandler.get_cognateset_batch(dev_dataset, langs, vocab, L2I, DEVICE)
    test_tensors = DataHandler.get_cognateset_batch(test_dataset, langs, vocab, L2I, DEVICE)

    model = Model(vocab,
                  num_layers=NUM_LAYERS,
                  dropout=DROPOUT,
                  feedforward_dim=FEEDFORWARD_DIM,
                  embedding_dim=EMBEDDING_SIZE,
                  model_size=HIDDEN_SIZE,
                  model_type=NETWORK,
                  langs=langs,
                  ).to(DEVICE)
    model.load_state_dict(saved_info['model'])
    loss_fn = torch.nn.CrossEntropyLoss()

    dev_loss, dev_ed, dev_feat_ed, dev_acc, dev_preds = evaluate(model, loss_fn, dev_tensors, DEVICE, MAX_LENGTH, vocab)
    test_loss, test_ed, test_feat_ed, test_acc, test_preds = evaluate(model, loss_fn, test_tensors, DEVICE, MAX_LENGTH, vocab)

    if not os.path.isdir('predictions'):
        os.mkdir('predictions')
    if not os.path.isdir('predictions/' + DATASET):
        os.mkdir('predictions/' + DATASET)
    write_preds('predictions/' + DATASET + '/best-' + criterion + '-dev', dev_preds)
    write_preds('predictions/' + DATASET + '/best-' + criterion + '-test', test_preds)

    # TODO: remember to calculate normalized edit distance
    print(f'===== <FINAL - best {criterion}>  (epoch: {saved_info["epoch"]}) ======')
    print(f'[dev]')
    print(f'  * loss: {dev_loss}')
    print(f'  * (character) edit distance: {dev_ed}')
    print(f'  * feature edit distance: {dev_feat_ed}')
    print(f'  * accuracy: {dev_acc}')
    print()
    print(f'[test]')
    print(f'  * loss: {test_loss}')
    print(f'  * (character) edit distance: {test_ed}')
    print(f'  * feature edit distance: {test_feat_ed}')
    print(f'  * accuracy: {test_acc}')

from main import *
import torch
import panphon.distance

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True,
                    help='chinese/romance_orthographic/romance_phonetic/austronesian')
args = parser.parse_args()
DATASET = args.dataset
MODELPATH_LOSS = f'./checkpoints/{DATASET}_best_loss.pt'
MODELPATH_ED = f'./checkpoints/{DATASET}_best_ed.pt'

# only call this function during final evaluation because calculating feature edit distance at every epoch could be costly
def get_metrics(model, loss_fn, dataset, device, max_length, vocab):
    model.eval()
    dist = panphon.distance.Distance()

    with torch.no_grad():
        total_loss = 0
        edit_distance = feature_edit_distance = phoneme_edit_distance = 0
        total_target_char_len, total_target_phoneme_len = 0, 0  # used to normalize the sum of the feature edit distances
        n_correct = 0
        predictions = []
        for _, (source_tokens, source_langs, target_tokens, target_langs) in dataset.items():
            # calculate loss
            logits = model(source_tokens, source_langs, target_tokens, target_langs, device)
            loss = loss_fn(logits, target_tokens)
            total_loss += loss.item()

            # calculate edit distance
            # necessary to have a separate encode and decode because we are doing greedy decoding here
            #   instead of comparing against the protoform
            (encoder_states, memory), embedded_x = model.encode(source_tokens, source_langs, device)
            prediction = model.decode(encoder_states, memory, embedded_x, max_length, device)

            # remove the BOS/EOS tokens to avoid deflating the normalized edit distance
            assert vocab.to_string(torch.unsqueeze(prediction[0], dim=0)) == '<' and vocab.to_string(torch.unsqueeze(target_tokens[0], dim=0)) == '<'
            assert vocab.to_string(torch.unsqueeze(prediction[-1], dim=0)) == '>' and vocab.to_string(torch.unsqueeze(target_tokens[-1], dim=0)) == '>'
            prediction = prediction[1:-1]
            target_tokens = target_tokens[1:-1]

            # TODO: get the batching correct
            predict_str, protoform_str = \
                vocab.to_string(prediction), vocab.to_string(target_tokens)

            # character edit distance
            edit_distance += get_edit_distance(predict_str, protoform_str)
            if predict_str == protoform_str:
                n_correct += 1
            predictions.append((predict_str, protoform_str))

            # feature edit distance
            feature_edit_distance += dist.feature_edit_distance(predict_str, protoform_str)

            # phoneme edit distance
            # keeping the prediction/target as a list preserves the phonemicization of the characters
            #   e.g. /th/ remains as one unit
            predicted_phonemes, gold_phonemes = vocab.to_string_list(prediction), vocab.to_string_list(target_tokens)
            phoneme_edit_distance += get_edit_distance(predicted_phonemes, gold_phonemes)

            total_target_char_len += len(protoform_str)
            total_target_phoneme_len += len(gold_phonemes)

    accuracy = n_correct / len(dataset)
    mean_loss = total_loss / len(dataset)

    # this is actually phoneme error rate because we group the phonemes together
    mean_edit_distance = edit_distance / len(dataset)
    # calculate with the reference - hypothesis could be arbitrarily long
    # aka character error rate
    character_error_rate = edit_distance / total_target_char_len

    # phoneme edit distance
    mean_phoneme_edit_distance = phoneme_edit_distance / len(dataset)
    # normalized phoneme edit distance, aka phoneme error rate
    phoneme_error_rate = phoneme_edit_distance / total_target_phoneme_len

    # normalize by # feats * # phonemes in longest sequence
    mean_feature_edit_distance = feature_edit_distance / len(dataset)
    feature_error_rate = dist.feature_error_rate([pred for (pred, _) in predictions], [hyp for (_, hyp) in predictions])

    return mean_loss, \
           mean_edit_distance, character_error_rate, \
           mean_phoneme_edit_distance, phoneme_error_rate, \
           mean_feature_edit_distance, feature_error_rate, \
           accuracy, predictions


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

    dev_loss, dev_ed, dev_cer, dev_phon_ed, dev_per, dev_feat_ed, dev_fer, dev_acc, dev_preds = get_metrics(model, loss_fn, dev_tensors, DEVICE, MAX_LENGTH, vocab)
    test_loss, test_ed, test_cer, test_phon_ed, test_per, test_feat_ed, test_fer, test_acc, test_preds = get_metrics(model, loss_fn, test_tensors, DEVICE, MAX_LENGTH, vocab)

    if not os.path.isdir('predictions'):
        os.mkdir('predictions')
    if not os.path.isdir('predictions/' + DATASET):
        os.mkdir('predictions/' + DATASET)
    write_preds('predictions/' + DATASET + '/best-' + criterion + '-dev', dev_preds)
    write_preds('predictions/' + DATASET + '/best-' + criterion + '-test', test_preds)

    print(f'===== <FINAL - best {criterion}>  (epoch: {saved_info["epoch"]}) ======')
    print(f'[dev]')
    print(f'  * loss: {dev_loss}')
    print(f'  * (character) edit distance: {dev_ed}')
    print(f'  * phoneme edit distance: {dev_phon_ed}')
    print(f'  * feature edit distance: {dev_feat_ed}')
    print(f'  * character error rate (CER): {dev_cer}')
    print(f'  * phoneme error rate (PER): {dev_per}')
    print(f'  * feature error rate (FER): {dev_fer}')
    print(f'  * accuracy: {dev_acc}')
    print()
    print(f'[test]')
    print(f'  * loss: {test_loss}')
    print(f'  * (character) edit distance: {test_ed}')
    print(f'  * phoneme edit distance: {test_phon_ed}')
    print(f'  * feature edit distance: {test_feat_ed}')
    print(f'  * character error rate (CER): {test_cer}')
    print(f'  * phoneme error rate (PER): {test_per}')
    print(f'  * feature error rate (FER): {test_fer}')
    print(f'  * accuracy: {test_acc}')

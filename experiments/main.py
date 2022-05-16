import argparse
from model import *
from tqdm import tqdm
import torch
import time
import numpy as np
import random
from preprocessing import DataHandler
import os
import wandb
import panphon


def get_edit_distance(s1, s2):
    # TODO: remove the BOS/EOS from consideration - affects normalized edit distance

    if len(s1) > len(s2):
        s1, s2 = s2, s1
    # len(s1) <= len(s2)
    # TODO: understand
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_

    return distances[-1]


def train_once(model, optimizer, loss_fn, train_data):
    model.train()  # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch

    order = list(train_data.keys())
    random.shuffle(order)
    good, bad = 0, 0
    total_train_loss = 0
    for cognate in order:
        source_tokens, source_langs, target_tokens, target_langs = train_data[cognate]
        # TODO: check if i'm supposed to do this
        optimizer.zero_grad()

        # TODO: do the to(device) thingy here
        logits = model(source_tokens, source_langs, target_tokens, target_langs, DEVICE)
        # logits should be (1, T, |Y|)

        # reshape logits to (T, |Y|) - remove batch dim for now
        # protoform_tensor: (T,)
        loss = loss_fn(logits, target_tokens)
        loss.backward()
        total_train_loss += loss.item()

        optimizer.step()

        # TODO: check dimensions for everything!

        # compare indices instead of converting to string
        predicted = torch.argmax(logits, dim=1)
        if torch.equal(predicted, target_tokens):
            good += 1
        else:
            bad += 1

    return total_train_loss / len(train_data), good / (good + bad)


def train(epochs, model, optimizer, loss_fn, train_data, dev_data):
    mean_train_losses, mean_dev_losses = np.zeros(epochs), np.zeros(epochs)
    best_loss_epoch, best_ed_epoch = 0, 0
    best_dev_loss, best_dev_edit_distance = 10e10, 10e10

    for epoch in tqdm(range(epochs)):
        t = time.time()

        train_loss, train_accuracy = train_once(model, optimizer, loss_fn, train_data)
        dev_loss, edit_distance, feat_ed, dev_accuracy, _ = evaluate(model, loss_fn, dev_data, DEVICE, MAX_LENGTH, vocab)
        wandb.log(
            {
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "dev_edit_distance": edit_distance,
                "dev_feat_edit_dist": feat_ed,
                "accuracy": dev_accuracy,
            }
        )
        print(f'< epoch {epoch} >  (elapsed: {time.time() - t:.2f}s)')
        print(f'  * [train]  loss: {train_loss:.6f}')
        dev_result_line = f'  * [ dev ]  loss: {dev_loss:.6f}'
        if edit_distance is not None:
            dev_result_line += f'  ||  edit distance: {edit_distance}  ||  accuracy: {dev_accuracy}'
        print(dev_result_line)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_loss_epoch = epoch
            save_model(model, optimizer, args, epoch, MODELPATH_LOSS)
        if edit_distance < best_dev_edit_distance:
            best_dev_edit_distance = edit_distance
            best_ed_epoch = epoch
            save_model(model, optimizer, args, epoch, MODELPATH_ED)
            wandb.log(
                {
                    "best_dev_edit_distance": best_dev_edit_distance,
                }
            )

        mean_train_losses[epoch] = train_loss
        mean_dev_losses[epoch] = dev_loss

    # TODO: be more specific in the naming
    if not os.path.isdir('losses'):
        os.mkdir('losses')
    if not os.path.isdir('losses/' + DATASET):
        os.mkdir('losses/' + DATASET)
    # TODO: the dynet loss values differ from the RNN's
    np.save(f"losses/{DATASET}/train", mean_train_losses)
    np.save(f"losses/{DATASET}/dev", mean_dev_losses)
    record(best_loss_epoch, best_dev_loss, best_ed_epoch, best_dev_edit_distance)

    # TODO: evaluate on test and record test


def record(best_loss_epoch, best_loss, best_ed_epoch, edit_distance):
    print(f'===== <TRAINING > ======')
    print(f'[dev]')
    print(f'  * loss: {best_loss}  (epoch: {best_loss_epoch})')
    print(f'  * edit distance: {edit_distance}  (epoch: {best_ed_epoch})')
    print()

    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir('results/' + DATASET):
        os.mkdir('results/' + DATASET)
    with open(f'./results/{DATASET}/params.txt', 'w') as fout:
        params = {'network': NETWORK,
                  'num_layers': NUM_LAYERS,
                  'model_size': HIDDEN_SIZE,
                  'lr': LEARNING_RATE,
                  'beta1': BETA_1,
                  'beta2': BETA_2,
                  'eps': EPS,
                  'embedding_size': EMBEDDING_SIZE,
                  'feedforward_dim': FEEDFORWARD_DIM,
                  'dropout': DROPOUT,
                  'epochs': NUM_EPOCHS,
                  'batch_size': 1}
        for k, v in params.items():
            fout.write(f'{k}: {v}\n')
    with open(f'./results/{DATASET}/metrics.txt', 'a') as fout:
        fout.write(f'{DATASET}. loss: {best_loss} ({best_loss_epoch})   ||   {edit_distance} ({best_ed_epoch})\n')


def evaluate(model, loss_fn, dataset, device, max_length, vocab):
    model.eval()
    dist = panphon.distance.Distance()

    with torch.no_grad():
        total_loss = 0
        edit_distance = 0
        feature_edit_distance = 0
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
            # TODO: get the batching correct
            predict_str, protoform_str = \
                vocab.to_string(prediction), vocab.to_string(target_tokens)
            edit_distance += get_edit_distance(predict_str, protoform_str)
            if predict_str == protoform_str:
                n_correct += 1
            predictions.append((predict_str, protoform_str))
            feature_edit_distance += dist.feature_edit_distance(predict_str, protoform_str)

    accuracy = n_correct / len(dataset)
    mean_loss = total_loss / len(dataset)
    mean_edit_distance = edit_distance / len(dataset)
    mean_feature_edit_distance = feature_edit_distance / len(dataset)

    return mean_loss, mean_edit_distance, mean_feature_edit_distance, accuracy, predictions


def save_model(model, optimizer, args, epoch, filepath):
    # TODO: is the model loading the parameters from the subclasses?
    save_info = {
        'model': model.state_dict(),
        # TODO: save torch.nn.CrossEntropyLoss()
        'optim': optimizer.state_dict(),
        'args': args,
        'epoch': epoch,
        'vocab': vocab,  # save the vocab to ensure the index mappings are the same during evaluation
        # langs does not need to be saved because it's an ordered list, not a set
    }
    torch.save(save_info, filepath)
    print(f'\t>> saved model to {filepath}')


def load_model(filepath):
    saved_info = torch.load(filepath)
    return saved_info


def write_preds(filepath, predictions):
    # predictions: predicted - original cognate
    # TODO: should we try adding the original cognate set
    with open(filepath, 'w') as f:
        f.write("prediction\tgold standard\n")
        for pred, gold_std in predictions:
            # remove BOS and EOS
            pred = pred[1:-1]
            gold_std = gold_std[1:-1]
            f.write(f"{pred}\t{gold_std}\n")


if __name__ == '__main__':
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='chinese/romance_orthographic/romance_phonetic/austronesian')
    parser.add_argument('--network', type=str, required=True, help='lstm/gru')
    parser.add_argument('--num_layers', type=int, required=True, help='number of RNN layers')
    parser.add_argument('--model_size', type=int, required=True, help='lstm hidden layer size')
    parser.add_argument('--lr', type=float, required=True, help='learning rate')
    parser.add_argument('--beta1', type=float, required=True, help='beta1')
    parser.add_argument('--beta2', type=float, required=True, help='beta2')
    parser.add_argument('--eps', type=float, required=True, help='eps')
    parser.add_argument('--embedding_size', type=int, required=True, help='embedding size')
    parser.add_argument('--feedforward_dim', type=int, required=True, help='dimension of the final MLP')
    parser.add_argument('--dropout', type=float, required=True, help='dropout value')
    parser.add_argument('--epochs', type=int, required=True)
    # TODO: batching
    parser.add_argument('--batch_size', type=int, required=True, help='batch_size')
    args = parser.parse_args()

    wandb.init(config=args)
    config = wandb.config

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = config["epochs"]  # args.epochs
    DATASET = args.dataset
    NUM_LAYERS = config["num_layers"]  # args.num_layers
    NETWORK = args.network
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    MODELPATH_LOSS = f'./checkpoints/{DATASET}_best_loss.pt'
    MODELPATH_ED = f'./checkpoints/{DATASET}_best_ed.pt'

    LEARNING_RATE = config["lr"]  # args.lr
    BETA_1 = config["beta1"]  # args.beta1
    BETA_2 = config["beta2"]  # args.beta2
    EPS = config["eps"]  # args.eps
    EMBEDDING_SIZE = config["embedding_size"]  # args.embedding_size
    DROPOUT = config["dropout"]  # args.dropout
    MAX_LENGTH = 30 if 'romance' in DATASET else 15
    HIDDEN_SIZE = config["model_size"]  # args.model_size
    FEEDFORWARD_DIM = config["feedforward_dim"]  # args.feedforward_dim

    train_dataset, phoneme_vocab, langs = DataHandler.load_dataset(f'./data/{DATASET}/train.pickle')
    dev_dataset, _, _ = DataHandler.load_dataset(f'./data/{DATASET}/dev.pickle')
    test_dataset, _, langs = DataHandler.load_dataset(f'./data/{DATASET}/test.pickle')
    # special tokens in the separator embedding's vocabulary
    langs = langs + ['sep']
    # TODO: create a special vocab just for the separator embeddings
    phoneme_vocab.add("<")
    phoneme_vocab.add(":")
    phoneme_vocab.add("*")
    phoneme_vocab.add(">")
    phoneme_vocab.add("<unk>")
    phoneme_vocab.add("-")
    phoneme_vocab.add("<s>")
    # treat each language as a token since each language will be included in the input sequence
    for lang in langs:
        phoneme_vocab.add(lang)

    vocab = Vocab(sorted(phoneme_vocab))
    L2I = {l: idx for idx, l in enumerate(langs)}

    model = Model(vocab,
                  num_layers=NUM_LAYERS,
                  dropout=DROPOUT,
                  feedforward_dim=FEEDFORWARD_DIM,
                  embedding_dim=EMBEDDING_SIZE,
                  model_size=HIDDEN_SIZE,
                  model_type=NETWORK,
                  langs=langs,
                  ).to(DEVICE)
    # note: Meloni et al do not do Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    # does the softmax for you
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 betas=(BETA_1, BETA_2),
                                 eps=EPS)

    # precompute the tensors once. reuse
    train_tensors = DataHandler.get_cognateset_batch(train_dataset, langs, vocab, L2I, DEVICE)
    dev_tensors = DataHandler.get_cognateset_batch(dev_dataset, langs, vocab, L2I, DEVICE)
    test_tensors = DataHandler.get_cognateset_batch(test_dataset, langs, vocab, L2I, DEVICE)
    train(NUM_EPOCHS, model, optimizer, loss_fn, train_tensors, dev_tensors)
    test_loss, test_ed, test_feat_ed, test_acc, test_preds = evaluate(model, loss_fn, test_tensors, DEVICE, MAX_LENGTH, vocab)


    # TODO: call the evaluate file. then report all of its summary metrics
        # report the best model's ED and loss
        # wandb.run.summary["test_ed"]

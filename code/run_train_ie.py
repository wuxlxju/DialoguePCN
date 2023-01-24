import os
import argparse
import numpy as np
import random
import time
import torch
import torch.nn
import torch.optim as optim
from dataloader import IEMOCAPDataset
from model import DialogueCRN
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from loss import FocalLoss
import logging
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

import datetime
now_time = datetime.datetime.now()
file_name = "/nfs/home/wuxl/DialoguePCN/results/"+str(now_time)+".log"
logger = get_logger(file_name)
logger.info('start training!')

def seed_everything(seed=2021):
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_loaders(path=None, valid_rate=0.1, batch_size=32, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(path=path, train=True)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_rate)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(path=path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_f, dataloader, epoch=0, train_flag=False, optimizer=None, cuda_flag=False, feature_type='text', target_names=None,
                        tensorboard=False):
    assert not train_flag or optimizer != None
    losses, preds, labels = [], [], []
    if train_flag:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train_flag: optimizer.zero_grad()
#110,32,100             110,32,512            110,32,100            110,32,2             32,110         32,110
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]
        # print('data',textf.shape, visuf.shape, acouf.shape, qmask.shape, umask.shape, label.shape)
        seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        if feature_type == 'multi':
            # (l,b,100+100+512)
            dataf = torch.cat((textf, acouf), dim=-1)
        else:
            dataf = textf if feature_type == 'text' else acouf


        log_prob = model(dataf, qmask, seq_lengths)#(110,32,100),(110,32,2),(32)
        #[1653,6]
        label = torch.cat([label[j][:seq_lengths[j]] for j in range(len(label))])#(32,110)
        loss = loss_f(log_prob, label)

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train_flag:
            loss.backward()
            if tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), float('nan'), [], []

    labels = np.array(labels)
    preds = np.array(preds)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    all_matrix = []
    all_matrix.append(metrics.classification_report(labels, preds, target_names=target_names, digits=4))
    all_matrix.append(["ACC"])
    for i in range(len(target_names)):
        all_matrix[-1].append("{}: {:.4f}".format(target_names[i], accuracy_score(labels[labels == i], preds[labels == i])))

    return avg_loss, avg_accuracy, avg_fscore, all_matrix, [labels, preds]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--status', type=str, default='train', help='optional status: train/test')

    parser.add_argument('--feature_type', type=str, default='multi', help='feature type multi/text/acouf')

    parser.add_argument('--data_dir', type=str, default='/nfs/home/wuxl/DialoguePCN/data/iemocap/IEMOCAP_features_bert.pkl', help='dataset dir')

    parser.add_argument('--output_dir', type=str, default='/nfs/home/wuxl/DialoguePCN/outputs/iemocap/dialoguecrn_v1', help='saved model dir')

    parser.add_argument('--load_model_state_dir', type=str, default='/nfs/home/wuxl/DialoguePCN/outputs/iemocap/dialoguecrn_v1/dialoguecrn_22_p5s6.pkl', help='load model state dir')

    parser.add_argument('--base_model', default='LSTM', help='base model, LSTM/GRU/Linear')

    parser.add_argument('--base_layer', type=int, default=2, help='the number of base model layers,1/2')

    parser.add_argument('--epochs', type=int, default=150, metavar='E', help='number of epochs')

    parser.add_argument('--patience', type=int, default=20, help='early stop')

    parser.add_argument('--batch_size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--valid_rate', type=float, default=0.0, metavar='valid_rate', help='valid rate, 0.0/0.1')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')#0.0001

    parser.add_argument('--l2', type=float, default=0.0002, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--dropout', type=float, default=0.2, metavar='dropout', help='dropout rate')

    parser.add_argument('--step_s', type=int, default=6, help='the number of reason turns at situation-level')#情景阶段推理的步数

    parser.add_argument('--step_p', type=int, default=5, help='the number of reason turns at speaker-level')#说话人阶段推理的步数

    parser.add_argument('--gamma', type=float, default=0, help='gamma 0/0.5/1/2/5')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--class_weight', action='store_true', default=True, help='use class weights')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='enables tensorboard log')

    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    args = parser.parse_args()
    print(args)

    epochs, batch_size, status, output_path, data_path, base_model, base_layer, feature_type = \
        args.epochs, args.batch_size, args.status, args.output_dir, args.data_dir, args.base_model, args.base_layer, args.feature_type
    cuda_flag = torch.cuda.is_available() and not args.no_cuda
    reason_steps = [args.step_s, args.step_p]

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    # IEMOCAP dataset
    n_classes, n_speakers, hidden_size, input_size = 6, 2, 100, None
    target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
    class_weights = torch.FloatTensor([1 / 0.087178797, 1 / 0.145836136, 1 / 0.229786089, 1 / 0.148392305, 1 / 0.140051123, 1 / 0.24875555])
    if feature_type == 'multi':
        input_size = 868
    elif feature_type in ['text', 'acouf']:
        input_size = 768
    else:
        print('Error: feature_type not set.')
        exit(0)

    seed_everything(seed=args.seed)
    model = DialogueCRN(base_model=base_model,
                        base_layer=base_layer,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        n_speakers=n_speakers,
                        n_classes=n_classes,
                        dropout=args.dropout,
                        cuda_flag=cuda_flag,
                        reason_steps=reason_steps)
    if cuda_flag:
        print('Running on GPU')
        class_weights = class_weights.cuda()
        model.cuda()
    else:
        print('Running on CPU')

    name = 'DialogueCRN_p5s6'
    print('{} with {} as base model'.format(name, base_model))
    print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print('Running on the {} features........'.format(feature_type))

    loss_f = FocalLoss(gamma=args.gamma, alpha=class_weights if args.class_weight else None)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(path=data_path, valid_rate=args.valid_rate, batch_size=batch_size, num_workers=0)

    if status == 'train':
        all_test_fscore, all_test_acc = [], []
        best_epoch, patience, best_eval_fscore, best_eval_loss = -1, 0, 0, None
        for e in range(epochs):
            start_time = time.time()

            train_loss, train_acc, train_fscore, _, _ = train_or_eval_model(model=model, loss_f=loss_f, dataloader=train_loader, epoch=e, train_flag=True,
                                                                            optimizer=optimizer, cuda_flag=cuda_flag, feature_type=feature_type,
                                                                            target_names=target_names)
            valid_loss, valid_acc, valid_fscore, _, _ = train_or_eval_model(model=model, loss_f=loss_f, dataloader=valid_loader, epoch=e, cuda_flag=cuda_flag,
                                                                            feature_type=feature_type, target_names=target_names)
            test_loss, test_acc, test_fscore, test_metrics, _ = train_or_eval_model(model=model, loss_f=loss_f, dataloader=test_loader, epoch=e,
                                                                                    cuda_flag=cuda_flag, feature_type=feature_type, target_names=target_names)
            all_test_fscore.append(test_fscore)
            all_test_acc.append(test_acc)

            if args.valid_rate > 0:
                eval_loss, _, eval_fscore = valid_loss, valid_acc, valid_fscore
            else:
                eval_loss, _, eval_fscore = test_loss, test_acc, test_fscore
            if e == 0 or best_eval_fscore < eval_fscore:
                best_epoch, best_eval_fscore = e, eval_fscore  
                if not os.path.exists(output_path): os.makedirs(output_path)
                save_model_dir = os.path.join(output_path, 'p5s6_{}_{}_{}.pkl'.format(name, e, best_eval_fscore).lower())
                if best_eval_fscore > 65.0:
                    torch.save(model.state_dict(), save_model_dir)
            if best_eval_loss is None:
                best_eval_loss = eval_loss
            else:
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    patience = 0
                else:
                    patience += 1

            if args.tensorboard:
                writer.add_scalar('train: accuracy/f1/loss', train_acc / train_fscore / train_loss, e)
                writer.add_scalar('valid: accuracy/f1/loss', valid_acc / valid_fscore / valid_loss, e)
                writer.add_scalar('test: accuracy/f1/loss', test_acc / test_fscore / test_loss, e)
                writer.close()

            print(
                'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                    format(e, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore,
                           round(time.time() - start_time, 2)))
            logger.info(
                'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                    format(e, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss,
                           test_acc, test_fscore,
                           round(time.time() - start_time, 2)))
            print(test_metrics[0])
            print(test_metrics[1])

            logger.info(test_metrics[0])
            logger.info(test_metrics[1])

            if patience >= args.patience:
                print('Early stoping...', patience)
                break

        print('Final Test performance...')
        print('Epoch: {}, Accuracy: {}, F1-Score: {}'.format(best_epoch,
                                                             all_test_acc[best_epoch] if best_epoch >= 0 else 0,
                                                             all_test_fscore[best_epoch] if best_epoch >= 0 else 0))

        logger.info('Epoch: {}, Accuracy: {}, F1-Score: {}'.format(best_epoch,
                                                             all_test_acc[best_epoch] if best_epoch >= 0 else 0,
                                                             all_test_fscore[best_epoch] if best_epoch >= 0 else 0))

    elif status == 'test':
        start_time = time.time()
        model.load_state_dict(torch.load(args.load_model_state_dir))
        test_loss, test_acc, test_fscore, test_metrics, test_outputs = train_or_eval_model(model=model, loss_f=loss_f, dataloader=test_loader,
                                                                                           cuda_flag=cuda_flag, feature_type=feature_type)

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc / test_loss, 0)
        print('test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.format(test_loss, test_acc, test_fscore, round(time.time() - start_time, 2)))
        print(test_metrics[0])
        print(test_metrics[1])

        logger.info('test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.format(test_loss, test_acc, test_fscore, round(time.time() - start_time, 2)))
        logger.info(test_metrics[0])
        logger.info(test_metrics[1])

    else:
        print('the status must be one of train/test')
        exit(0)

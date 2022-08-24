import argparse
# import logging
import random

from collections import OrderedDict

import numpy as np
import torch
from sklearn.svm import LinearSVC, SVC
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from torch_scatter import scatter
from datasets import TUDataset, TUEvaluator
from LGA_Lib.embedding_evaluation import EmbeddingEvaluation
from LGA_Lib.encoder import TUEncoder
from LGA_Lib.encoder import TUEncoder_sd
from LGA_Lib.learning import MModel
from LGA_Lib.learning import MModel_sd
from LGA_Lib.utils import initialize_edge_weight, initialize_node_features, set_tu_dataset_y_shape
from LGA_Lib.LGA_learner import LGALearner
import warnings
import time


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)




def run(args):
    warnings.filterwarnings('ignore')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    setup_seed(args.seed)

    evaluator = TUEvaluator()
    my_transforms = Compose([initialize_node_features, initialize_edge_weight, set_tu_dataset_y_shape])
    dataset = TUDataset("./original_datasets/", args.dataset, transform=my_transforms)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MModel(
        TUEncoder(num_dataset_features=1, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
        args.emb_dim).to(device)

    model_sd = MModel_sd(
        TUEncoder_sd(num_dataset_features=1, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
        args.emb_dim, device=device).to(device)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)

    LGA_learner = LGALearner(TUEncoder(num_dataset_features=1, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
                               mlp_edge_model_dim=args.mlp_edge_model_dim).to(device)
    LGA_optimizer = torch.optim.Adam(LGA_learner.parameters(), lr=args.LGA_lr)
    if args.downstream_classifier == "linear":
        ee = EmbeddingEvaluation(LinearSVC(dual=False, fit_intercept=True), evaluator, dataset.task_type, dataset.num_tasks,
                             device, param_search=True)
    else:
        ee = EmbeddingEvaluation(SVC(), evaluator, dataset.task_type,
                                 dataset.num_tasks,
                                 device, param_search=True)

    model.eval()
    train_score, val_score, test_score = ee.kf_embedding_evaluation(model.encoder, dataset)
    print("Performance Before training: Train: {} Val: {} Test: {}".format(train_score, val_score,
                                                                                          test_score))


    model_losses = []
    LGA_losses = []
    LGA_regs = []
    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        model_loss_all = 0
        LGA_loss_all = 0
        reg_all = 0
        for batch in dataloader:
            # set up
            batch = batch.to(device)


            # ========================train model======================== #
            model.train()
            LGA_learner.eval()
            model.zero_grad()

            x, _ = model(batch.batch, batch.x, batch.edge_index, None, None)
            edge_logits = LGA_learner(batch.batch, batch.x, batch.edge_index, None)

            bias = 0.0001
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            edge_score = torch.log(eps) - torch.log(1 - eps)
            edge_score = edge_score.to(device)
            edge_score = (edge_score + edge_logits)
            batch_aug_edge_weight = torch.sigmoid(edge_score).squeeze().detach()

            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, batch_aug_edge_weight)

            model_loss = model.calc_loss(x, x_aug)
            model_loss_all += model_loss.item() * batch.num_graphs
            model_loss.backward()
            model_optimizer.step()

            # ========================train LGA======================== #
            LGA_learner.train()
            LGA_learner.zero_grad()
            model.eval()

            x, _ = model(batch.batch, batch.x, batch.edge_index, None, None)

            edge_logits = LGA_learner(batch.batch, batch.x, batch.edge_index, None)


            bias = 0.0001
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            edge_score = torch.log(eps) - torch.log(1 - eps)
            edge_score = edge_score.to(device)
            edge_score = (edge_score + edge_logits)
            batch_aug_edge_weight = torch.sigmoid(edge_score).squeeze()

            row, col = batch.edge_index
            edge_batch = batch.batch[row]

            uni, edge_batch_num = edge_batch.unique(return_counts=True)
            sum_pe = scatter((1 - batch_aug_edge_weight), edge_batch, reduce="sum")

            reg = []
            for b_id in range(args.batch_size):
                if b_id in uni:
                    num_edges = edge_batch_num[uni.tolist().index(b_id)]
                    reg.append(sum_pe[b_id] / num_edges)
                else:
                    pass
            reg = torch.stack(reg)
            reg = reg.mean()
            ratio = reg / args.reg_expect

            batch_aug_edge_weight = batch_aug_edge_weight / ratio # edge weight generalization

            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, batch_aug_edge_weight)

            model_loss = model.calc_loss(x, x_aug)

            # current parameter
            fast_weights = OrderedDict((name, param) for (name, param) in model.named_parameters())

            # create_graph flag for computing second-derivative
            grads = torch.autograd.grad(model_loss, model.parameters(), create_graph=True)
            data = [p.data for p in list(model.parameters())]

            # compute parameter' by applying sgd on multi-task loss
            fast_weights = OrderedDict(
                (name, param - args.LGA_lr * grad) for ((name, param), grad, data)
                                                         in zip(fast_weights.items(), grads, data))

            # compute primary loss with the updated parameter'
            x, _ = model_sd.forward(batch.batch, batch.x, batch.edge_index, None, None, weights=fast_weights)
            x_aug, _ = model_sd.forward(batch.batch, batch.x, batch.edge_index, None, batch_aug_edge_weight.detach(),
                                        weights=fast_weights)
            LGA_loss = 0.1 * model.calc_feature_loss(x, x_aug) + model.calc_instance_loss(x, x_aug)

            LGA_loss_all += LGA_loss.item() * batch.num_graphs
            reg_all += reg.item()
            LGA_loss.backward()
            LGA_optimizer.step()

        fin_model_loss = model_loss_all / len(dataloader)
        fin_LGA_loss = LGA_loss_all / len(dataloader)
        fin_reg = reg_all / len(dataloader)

        print('Epoch {}, Model Loss {}, LGA Loss {}'.format(epoch, fin_model_loss, fin_LGA_loss))
        model_losses.append(fin_model_loss)
        LGA_losses.append(fin_LGA_loss)
        LGA_regs.append(fin_reg)
        if epoch % args.eval_interval == 0:
            model.eval()

            train_score, val_score, test_score = ee.kf_embedding_evaluation(model.encoder, dataset)

            print("Metric: {} Train: {} Val: {} Test: {}".format(evaluator.eval_metric, train_score,
                                                                 val_score, test_score))

            print("Epoch " , epoch , "Train", train_score, "Val", val_score, "Test", test_score)

            train_curve.append(train_score)
            valid_curve.append(val_score)
            test_curve.append(test_score)

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))

    return valid_curve[best_val_epoch], test_curve[best_val_epoch]


def arg_parse():
    parser = argparse.ArgumentParser(description='MEGA Training')

    parser.add_argument('--dataset', type=str, default='IMDB-BINARY',
                        help='Dataset')
    parser.add_argument('--model_lr', type=float, default=0.001,
                        help='Model Learning rate.')
    parser.add_argument('--LGA_lr', type=float, default=0.0001,
                        help='LGA Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=3,
                        help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='standard',
                        help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.0,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--emb_dim', type=int, default=32,
                        help='embedding dimension')
    parser.add_argument('--mlp_edge_model_dim', type=int, default=32,
                        help='embedding dimension')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Train Epochs')
    parser.add_argument('--eval_interval', type=int, default=5, help="eval epochs interval")
    parser.add_argument('--downstream_classifier', type=str, default="linear", help="Downstream classifier is linear or non-linear")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--weight_on_diag', type=float, default=1.0)
    parser.add_argument('--max_sd_epoch', type=int, default=50)
    parser.add_argument('--weight_on_instance_diag', type=float, default=1.0)
    parser.add_argument('--min_weight', type=float, default=0.0)
    parser.add_argument('--reg_expect', type=float, default=0.4)

    return parser.parse_args()


if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    args = arg_parse()

    val, test = run(args)

    print("===================TEST RESULT====================")
    print("val result is ", val, "\n test result is", test)


    warnings.filterwarnings('ignore')


    args = arg_parse()

    counter = 0
    test_max = 0
    test_min = 100000000
    test_sum = 0

    for i in range(10):
        val, test=run(args)
        print("val is",val,"test is",test)
        if test_max < test:
            test_max = test
        if test_min > test:
            test_min = test
        test_sum = test_sum + test
        test_average = test_sum/(i+1)
        print("=============================")
        print("=========round is",i,"=======")
        print("==test_min is ",test_min,"==")
        print("==test_max is ", test_max, "==")
        print("==test_average is ", test_average, "==")
        print("=============================")
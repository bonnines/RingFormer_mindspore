import warnings
import deepchem as dc
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR

from data_loader_het import  get_dataset_het
from model_het import HeteroTransformer
from copy import deepcopy
import  torch_geometric.transforms as T 
from transform import AddHetRingDegreePE, AddHetRingPathPE, AddRingRandomWalkPE, AddVirtualMol, AddHetRingTypeDegreePE, DelHetEdgeType


import random
import numpy as np
import torch
import torch_geometric
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch_geometric.seed_everything(seed)
    
    
    
    




def train(args, filename=None):
    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')
        
    maes, mapes, mses = [], [], []
    best_vals = []

    float_pe = False
    pe_dim = 7
    num_ring_edge_types = 1
    add_mol =False

    transform = T.Compose([AddHetRingDegreePE(pe_dim), AddVirtualMol()])
    add_mol = True

    # if args.use_old_data:
    dataloader,  dataloader_test, dataloader_val, transformer, meta = get_dataset_het(args, transform)
    # else:
    #     dataloader,  dataloader_test, dataloader_val, transformer, meta = get_dataset_het(args, transform)
    num_classes = meta['num_classes']
    n_train = len(dataloader.dataset)
    n_val = len(dataloader_val.dataset)
    n_test = len(dataloader_test.dataset) 
        
    for trial in range(args.num_trial):
        setup_seed(trial)  
        # Model initialization          
        model = HeteroTransformer(dataloader.dataset[0].metadata(), num_classes, args.hidden_dim,args.num_layer, heads=args.heads, conv=args.model, ring_conv=args.ring_conv, pool=args.pool, norm = 'BatchNorm' 
                    if args.bn else args.norm, transformer_norm=args.transformer_norm, l2norm=args.l2norm, dropout=args.dropout, attn_dropout=args.attn_dropout, criterion = args.criterion, jk=args.jk, final_jk = args.final_jk, aggr=args.aggr, 
                    normalize=args.normalize, first_residual=args.first_residual, residual=args.residual, ring_init=args.ring_init, pe_dim=pe_dim, cat_pe=args.cat_pe, use_bias=args.use_bias, 
                    add_mol=add_mol, combine_mol=args.combine_mol, float_pe=float_pe, combine_edge=args.combine_edge, root_weight=args.root_weight, num_ring_edge_types=num_ring_edge_types, 
                    clip_attn=args.clip_attn, model='Transformer').to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.scheduler.startswith('step'):
            step_size, gamma = args.scheduler.split('-')[1:]
            scheduler = StepLR(optimizer, step_size=int(step_size), gamma=float(gamma))
        elif args.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epoch) 
        elif args.scheduler.startswith('onecycle'):
            pct_start = float(args.scheduler.split('-')[1]) if '-' in args.scheduler else 0.1
            scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(dataloader), epochs=args.num_epoch, pct_start=pct_start)
        else:
            scheduler = None
        # Training&Validation
        best_val = float("Inf")
        best_epoch = 0
        for epoch in range(1, args.num_epoch + 1):
            model.train()
            loss_all = 0
            for data in dataloader:
                data = data.to(device)
                optimizer.zero_grad()
                loss = model.calc_loss(data)
                loss_all += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            print('[TRAIN] Epoch:{:03d} | Loss:{:.4f}'.format(epoch, loss_all / n_train))
            # Validation
            model.eval()
            loss_all_val = 0.0                
            with torch.no_grad():            
                for data in dataloader_val:
                    data = data.to(device)
                    loss = model.calc_loss(data)
                    loss_all_val += loss.item() * data.num_graphs
            if loss_all_val < best_val:
                best_val = loss_all_val
                best_model = deepcopy(model.state_dict())
                best_epoch = epoch  

            if epoch % args.eval_freq == 0:
                model.eval()
                y_true = []
                y_preds = []                   
                with torch.no_grad():
                    for data in dataloader_test:
                        data = data.to(device)
                        y_true = y_true + data.y.cpu().reshape(-1, num_classes).tolist()
                        y_preds = y_preds + model.predict_score(data).cpu().reshape(-1, num_classes).tolist()
                y_true = torch.Tensor(y_true)
                y_preds = torch.Tensor(y_preds)
                test_mask = y_true != 0
                y_true = y_true[test_mask].reshape(-1,1).tolist()
                y_preds = y_preds[test_mask].reshape(-1,1).tolist()                           
                if args.normalize:
                    y_true= transformer.inverse_transform(y_true)
                    y_preds = transformer.inverse_transform(y_preds) 
                mae = mean_absolute_error(y_true, y_preds)
                mape = mean_absolute_percentage_error(y_true, y_preds)
                mse = mean_squared_error(y_true, y_preds)                                   
        # Test on best validation
        model.load_state_dict(best_model) 
        model.eval()
        y_true = []
        y_preds = []
        with torch.no_grad():     
            for data in dataloader_test:
                data = data.to(device)
                y_true = y_true + data.y.cpu().reshape(-1, num_classes).tolist()
                y_preds = y_preds + model.predict_score(data).cpu().reshape(-1, num_classes).tolist()
        assert len(y_true) == n_test and len(y_preds) == n_test
        y_true = torch.Tensor(y_true)
        y_preds = torch.Tensor(y_preds)
        test_mask = y_true != 0
        y_true = y_true[test_mask].reshape(-1,1).tolist()
        y_preds = y_preds[test_mask].reshape(-1,1).tolist()            
        if args.normalize:
            y_true= transformer.inverse_transform(y_true)
            y_preds = transformer.inverse_transform(y_preds)                     
        mae = mean_absolute_error(y_true, y_preds)
        mape = mean_absolute_percentage_error(y_true, y_preds)
        mse = mean_squared_error(y_true, y_preds)     

        maes.append(mae)
        mapes.append(mape)
        mses.append(mse)

        best_vals.append(best_val)

    # Print average results  
    
    avg_val = np.mean(maes)
    std_val = np.std(maes)
    print('MAE: {:.4f}+-{:.4f}'.format(avg_val, std_val))   
    
    
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='HOPV', choices=['HOPV', 'PFD', 'NFA', 'PD', 'CEPDB'])
    parser.add_argument('-dataset_version', type=str, default='V1')
    parser.add_argument('-featurizer', type=str, default=None, choices=[None, 'MACCS', 'ECFP6', 'Mordred'])
    parser.add_argument('-normalize', type=bool, default=False)
    parser.add_argument('-scaler', type=str, default='standard', choices=['minmax', 'standard'])
    parser.add_argument('-frac_train', type=float, default=0.6)
    parser.add_argument('-target_mode', type=str, default='single')
    parser.add_argument('-target_task', type=int, default=0, help='0: PCE, 1: HOMO, 2: LUMO, 3: band gap, 4: Voc, 5: Jsc, 6: FF')
    parser.add_argument('-splitter', type=str, default='scaffold')
    parser.add_argument('-model', type=str, default='GINE')
    parser.add_argument('-ring_conv', type=str, default='SparseEdge')
    parser.add_argument('-num_trial', type=int, default=1)
    parser.add_argument('-gpu', type=int, default=1)

    parser.add_argument('-num_epoch', type=int, default=10)
    parser.add_argument('-eval_freq', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-bn', type=bool, default=False)
    parser.add_argument('-norm', type=str, default=None, choices=[None, 'BatchNorm', 'LayerNorm'])
    parser.add_argument('-transformer_norm', type=str, default='LayerNorm', choices=[None, 'BatchNorm', 'LayerNorm'])


    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-weight_decay', type=float, default=5e-4)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-attn_dropout', type=float, default=0.0)
    parser.add_argument('-criterion', type=str, default='MAE')
    parser.add_argument('-scheduler', type=str, default='onecycle-0.05')

    parser.add_argument('-num_layer', type=int, default=5)
    parser.add_argument('-hidden_dim', type=int, default=128)
    parser.add_argument('-heads', type=int, default=4)
    parser.add_argument('-l2norm', type=bool, default=False)
    parser.add_argument('-pool', type=str, default='add')
    parser.add_argument('-jk', type=str, default='cat')
    parser.add_argument('-final_jk', type=str, default='cat')
    parser.add_argument('-aggr', type=str, default='cat')
    parser.add_argument('-ring_init', type=str, default='random')
    parser.add_argument('-first_residual', type=bool, default=True)
    parser.add_argument('-residual', type=bool, default=True)
    parser.add_argument('-use_bias', type=bool, default=False)

    parser.add_argument('-transform', type=str, default=None, choices=[None, 'VirtualNode'])
    parser.add_argument('-best_val', type=bool, default=True)

    parser.add_argument('-PE', type=str, default='RingDegree', choices=['RingDegree', 'RingBondDegree', 'RandomWalk']) # 
    parser.add_argument('-pe_dim', type=int, default=7)
    parser.add_argument('-cat_pe', type=bool, default=True)

    parser.add_argument('-combine_mol', type=str, default='add')
    parser.add_argument('-root_weight', type=bool, default=True)
    parser.add_argument('-combine_edge', type=str, default='cat', choices=['add', 'add_lin', 'cat','add_lin'])
    parser.add_argument('-clip_attn', type=bool, default=True)
    parser.add_argument('-add_cross', type=bool, default=True)
    parser.add_argument('-add_mol', type=bool, default=True)

    lr_list_dict = {'HOPV': [5e-4, 1e-3], 'PolymerFA': [0.001, 0.0005, 1e-4, 5e-5], 'pNFA': [0.001,  5e-4, 1e-4, 5e-5], 'nNFA': [0.0001, 5e-5]}
    args = parser.parse_args()                
    train(args, None)


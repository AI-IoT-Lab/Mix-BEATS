from data.data_factory import data_provider
from models import MixBEATS, NBEATS, Informer, Autoformer, iTransformer, Reformer, DLinear, PatchTST, TimesNet, FEDformer, Transformer, Pyraformer, FreTS, TSMixer, LightTS, SegRNN, TiDE, SCINet
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
from torch import optim
import json
from time import time
from tqdm import tqdm
import argparse

import os

import warnings                                                                                     
import numpy as np
warnings.filterwarnings('ignore')


import random
fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
                                                                                                                                                                                                                                                              

def test(args, model, criterion, device, test_loader):


    # Test phase
    model.eval()
    test_losses = []
    trues = []
    preds = []

    count = 0

    if not os.path.exists(args["result_path"]):
            os.makedirs(args["result_path"])

        
    for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        with torch.no_grad():
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args["pred_len"]:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args["label_len"], :], dec_inp], dim=1).float().to(device)
            # encoder - decoder

            def _run_model():
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if args["output_attention"]:
                    outputs = outputs[0]
                return outputs

            if args["use_amp"]:
                with torch.amp.autocast():
                    outputs = _run_model()
            else:
                outputs = _run_model()

            f_dim = -1 if args["features"] == 'MS' else 0
            outputs = outputs[:, -args["pred_len"]:, f_dim:]
            batch_y = batch_y[:, -args["pred_len"]:, f_dim:].to(device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            loss = criterion(pred, true)

            pred = pred.numpy()
            true = true.numpy()

            preds.append(pred)
            trues.append(true)

            if count % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                mae, mse, rmse, mape, mspe, nrmse = metric(pred[0, :, -1], true[0, :, -1])
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

                plot_path = os.path.join(args["result_path"], 'plots')
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)

                seq_len = input.shape[1]
                pred_len = pred.shape[1]

                visual(gt, seq_len, pred_len, mse, pd, os.path.join(plot_path, str(count) + '.pdf'))

            test_losses.append(loss)

            count += 1


    # Calculate average validation loss and RMSE
    avg_test_loss = np.mean(test_losses)
    # print("Test loss is: ",avg_test_loss)


    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    # print('test shape:', preds.shape, trues.shape)

    return preds, trues   



if __name__ == '__main__':   

    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    parser.add_argument('--config-file', type=str, default='./configs/ETTh1/ETTh1_96_24.json', help='Input config file path', required=True)

    # Optional overrides
    parser.add_argument('--model', type=str, default='MixBEATS', help='Enter model name')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints/MixBEATS/ETTh1_96_24', help='Path to save model')
    parser.add_argument('--result_path', type=str, default='./results/MixBEATS/ETTh1_96_24', help='Path to save results')

    # Parse known args
    cli_args = parser.parse_args()

    # Load config file
    with open(cli_args.config_file, 'r') as f:
        args = json.load(f)

    args['model'] = cli_args.model
    args['model_save_path'] = cli_args.model_save_path
    args['result_path'] = cli_args.result_path


    if args['model'] == 'TimesNet':
        args['d_model'] = 16
        args['d_ff'] = 32  

    if args['model'] == 'MixBEATS':
        if 'illness' in args['data_path']:
            args['patch_size'] = 18


    test_dataset, test_loader = data_provider(args, flag='test')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model_dict = {
                'Autoformer': Autoformer,
                'iTransformer': iTransformer,
                'Informer': Informer,
                'Reformer': Reformer,
                'MixBEATS': MixBEATS,
                'NBEATS': NBEATS,
                'DLinear': DLinear,
                'FEDformer': FEDformer,
                'PatchTST': PatchTST,
                'TimesNet': TimesNet,
                'Transformer': Transformer,
                'Pyraformer': Pyraformer,
                'FreTS': FreTS,
                'TSMixer': TSMixer,
                'LightTS': LightTS,
                'SegRNN': SegRNN,
                'TiDE': TiDE,
                'SCINet': SCINet,
    }
    model = model_dict[args["model"]].Model(args).float()

    if args["use_multi_gpu"] and args["use_gpu"]:
        model = torch.nn.DataParallel(model, device_ids=args["device_ids"])

    model = model.to(device)
    model.load_state_dict(torch.load(f'{args["model_save_path"]}/best_model.pth'))

    optimizer = optim.Adam(model.parameters(), lr=args["learning_rate"])
    if args["loss"] == "mse":
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.HuberLoss(reduction="mean", delta=1.0)


    start_time = time()
    preds, trues = test(args, model, criterion, device, test_loader)

    mae, mse, rmse, mape, mspe, nrmse = metric(preds, trues)

    end_time = time() - start_time


    # print('dataset:{}, mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, nrmse:{}, inference time:{}'.format(args["model_id"], mse, mae, rmse, mape, mspe, nrmse, end_time))
    id = args['model']+'_'+args['model_id'][9:]
    print(f'{id},{mse:.3f},{mae:.3f},{rmse:.3f},{mape:.3f},{mspe:.3f},{nrmse:.3f},{end_time:.3f}')

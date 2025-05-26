from data.data_factory import data_provider
from models import MixBEATS, NBEATS, Informer, Autoformer, iTransformer, Reformer, DLinear, PatchTST, TimesNet, FEDformer, Transformer, Pyraformer, FreTS, TSMixer, LightTS, SegRNN, TiDE, SCINet, GridFlow
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

                                                                                                                                                                                                                                                                                      

def train(args, model, criterion, optimizer, device, train_loader, val_loader):

    # Early stopping parameters
    patience = args["patience"]
    best_val_loss = float('inf')                                                                                                                                                                                 
    counter = 0
    early_stop = False

    if args["use_amp"]:
        scaler = torch.amp.GradScaler()

    num_epochs = args["num_epochs"]
    train_start_time = time()  # Start timer

    for epoch in range(num_epochs):

        if early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break  

        model.train()
        train_losses = []

        epoch_start_time = time()  # Start epoch timer

        # Progress bar for the training loop
        with tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}', leave=False) as pbar:
            for batch_x, batch_y, batch_x_mark, batch_y_mark in pbar:

                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                
                optimizer.zero_grad()

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
                
                loss = criterion(outputs, batch_y)
                train_losses.append(loss.item())

                if args["use_amp"]:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                pbar.set_postfix(loss=loss.item(), elapsed=f"{time() - epoch_start_time:.2f}s")
        
        # Calculate average training loss
        avg_train_loss = np.mean(train_losses)

        # Validation phase
        model.eval()
        val_losses = []
        y_true_val = []
        y_pred_val = []

        # Progress bar for the validation loop
        with tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{num_epochs}', leave=False) as pbar:
            for batch_x, batch_y, batch_x_mark, batch_y_mark in pbar:
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

                    val_losses.append(loss)

        # Calculate average validation loss and RMSE
        avg_val_loss = np.mean(val_losses)

        # Print epoch summary
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            # Save the best model parameters
            os.makedirs(args["model_save_path"], exist_ok=True)
            torch.save(model.state_dict(), f'{args["model_save_path"]}/best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                early_stop = True


        adjust_learning_rate(optimizer, epoch + 1, args)


    total_training_time = time() - train_start_time
    print(f'Total Training Time: {total_training_time:.2f}s')







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    parser.add_argument('--config-file', type=str, default='./configs/ETTh1/ETTh1_96_24.json', help='Input config file path', required=True)

    # Optional overrides
    parser.add_argument('--model', type=str, default='MixBEATS', help='Enter model name')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints/MixBEATS/ETTh1_96_24', help='Path to save model')

    # Parse known args
    cli_args = parser.parse_args()

    # Load config file
    with open(cli_args.config_file, 'r') as f:
        args = json.load(f)

    args['model'] = cli_args.model
    args['model_save_path'] = cli_args.model_save_path

    if args['model'] == 'TimesNet':
        args['d_model'] = 16
        args['d_ff'] = 32  

    if args['model'] == 'MixBEATS':
        if 'illness' in args['data_path']:
            args['patch_size'] = 18




    if args['model'] == 'GridFlow':
        train_data, train_loader = data_provider_gridflow(args, flag='train')
        val_data, val_loader = data_provider_gridflow(args, flag='val')
    else:
        train_data, train_loader = data_provider(args, flag='train')
        val_data, val_loader = data_provider(args, flag='val')

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
                'SCINet': SCINet,
                'TiDE': TiDE,
                'GridFlow': GridFlow,
    }
    model = model_dict[args["model"]].Model(args).float()

    if args["use_multi_gpu"] and args["use_gpu"]:
        model = torch.nn.DataParallel(model, device_ids=args["device_ids"])

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args["learning_rate"])
    if args["loss"] == "mse":
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.HuberLoss(reduction="mean", delta=1.0)

    if args['model'] == 'GridFlow':
        train_gridflow(args, model, criterion, optimizer, device, train_loader, val_loader)
    else:
        train(args, model, criterion, optimizer, device, train_loader, val_loader)



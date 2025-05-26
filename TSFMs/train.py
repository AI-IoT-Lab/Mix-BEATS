import numpy as np
import pandas as pd
import argparse
import json
import os
import sys
sys.path.append('./model')

import torch
from torch.utils.data import Dataset, DataLoader
from models import MixBEATS
from torch.utils.data import ConcatDataset

from tqdm import tqdm
from time import time
from sklearn.metrics import mean_squared_error
from utils.tools import EarlyStopping, adjust_learning_rate, visual



def standardize_series(series, eps=1e-8):
    mean = np.mean(series)
    std = np.std(series)
    standardized_series = (series - mean) / (std+eps)
    return standardized_series, mean, std

def unscale_predictions(predictions, mean, std, eps=1e-8):
    return predictions * (std+eps) + mean


class TimeSeriesDataset(Dataset):
    def __init__(self, data, backcast_length, forecast_length, stride=1):
        # Standardize the time series data
        self.data, self.mean, self.std = standardize_series(data)
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stride = stride

    def __len__(self):
        return (len(self.data) - self.backcast_length - self.forecast_length) // self.stride + 1

    def __getitem__(self, index):
        start_index = index * self.stride
        x = self.data[start_index : start_index + self.backcast_length]
        y = self.data[start_index + self.backcast_length : start_index + self.backcast_length + self.forecast_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_datasets(folder_path, backcast_length, forecast_length, stride):
    datasets = []

    for region in os.listdir(folder_path):

        region_path = os.path.join(folder_path, region)
        for building in os.listdir(region_path):

            if building.endswith('.csv'):
                file_path = os.path.join(region_path, building)
                df = pd.read_csv(file_path)
                energy_data = df['energy'].values
                dataset = TimeSeriesDataset(energy_data, backcast_length, forecast_length, stride)
                datasets.append(dataset)

    combined_dataset = ConcatDataset(datasets)
    return combined_dataset






def train(args, model, criterion, optimizer, device, train_loader, val_loader, param):

    # Early stopping parameters
    patience = args['patience']
    best_val_loss = float('inf')
    counter = 0
    early_stop = False

    num_epochs = args["num_epochs"]
    train_start_time = time()  # Start timer 

    t_loss = []
    v_loss = []

    for epoch in range(num_epochs):

        if early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break  

        model.train()
        train_losses = []

        epoch_start_time = time()  # Start epoch timer

        # Progress bar for the training loop
        with tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}', leave=False) as pbar:
            for x_batch, y_batch in pbar:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                backcast, forecast = model(x_batch)
                loss = criterion(forecast, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                pbar.set_postfix(loss=loss.item(), elapsed=f"{time() - epoch_start_time:.2f}s")
        
        # Calculate average training loss
        avg_train_loss = np.mean(train_losses)

        t_loss.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_losses = []
        y_true_val = []
        y_pred_val = []

        # Progress bar for the validation loop
        with tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{num_epochs}', leave=False) as pbar:
            for x_val, y_val in pbar:
                x_val, y_val = x_val.to(device), y_val.to(device)
                with torch.no_grad():
                    backcast, forecast = model(x_val)
                    loss = criterion(forecast, y_val)
                    val_losses.append(loss.item())
                    
                    # Collect true and predicted values for RMSE calculation
                    y_true_val.extend(y_val.cpu().numpy())
                    y_pred_val.extend(forecast.cpu().numpy())

        # Calculate average validation loss and RMSE
        avg_val_loss = np.mean(val_losses)
        v_loss.append(avg_val_loss)

        rmse_val = np.sqrt(mean_squared_error(y_true_val, y_pred_val))

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

    loss_data = {
        "param": param,
        "train_loss": t_loss,
        "val_loss": v_loss
    }



    loss_data_path = f'{args["model_save_path"]}/loss_data.json'
    with open(loss_data_path, "w") as f:
        json.dump(loss_data, f)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    parser.add_argument('--config-file', type=str, default='./configs/energy_data.json', help='Input config file path', required=True)
    parser.add_argument('--seq_len', type=int, default=168, help='Input Sequence Length')
    parser.add_argument('--stride', type=int, default=24, help='Input Stride')
    parser.add_argument('--patch_size', type=int, default=24, help='Input Patch Length')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Input Hidden Dim')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints/MixBEATS')
    parser.add_argument('--num_epochs', type=int, default=100, help='Enter the number of epochs')

    
    # Parse known args
    cli_args = parser.parse_args()

    # Load config file
    with open(cli_args.config_file, 'r') as f:
        args = json.load(f)


    args['model_save_path'] = cli_args.model_save_path
    args['seq_len'] = cli_args.seq_len
    args['stride'] = cli_args.stride 
    args['hidden_dim'] = cli_args.hidden_dim
    args['patch_size'] = cli_args.patch_size
    args['num_epochs'] = cli_args.num_epochs


    
    # # Parameters
    backcast_length = args['seq_len']
    forecast_length = args['pred_len']
    stride = args['stride']
    batch_size = args['batch_size']
    patch_size = args['patch_size']
    hidden_dim = args['hidden_dim']
    num_patches = backcast_length // patch_size

    # Load datasets
    train_datasets = load_datasets(args['train_dataset_path'], backcast_length, forecast_length, stride)
    val_datasets = load_datasets(args['val_dataset_path'], backcast_length, forecast_length, stride)

    # Create data loaders
    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_datasets, batch_size=batch_size, shuffle=True)

    

    # check device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define N-BEATS model
    model = MixBEATS.Model(
        device=device,
        forecast_length=forecast_length,
        backcast_length=backcast_length,
        patch_size = patch_size, 
        num_patches = num_patches,
        hidden_dim=hidden_dim,
    ).to(device)

    # model's parameters
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model's parameter count is:", param)



    # Define loss and optimizer
    if args['loss'] == 'mse':
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.HuberLoss(reduction="mean", delta=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])

    # training the model and save best parameters
    train(args, model, criterion, optimizer, device, train_loader, val_loader, param)

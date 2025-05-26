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
from utils.metrics import cal_cvrmse, cal_mae, cal_mse, cal_nrmse


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



def test(args, model, criterion, device):

    folder_path = args['test_dataset_path']
    result_path = args['result_path']
    backcast_length = args['seq_len']
    forecast_length = args['pred_len']
    stride = args['stride']


    median_res = []  
    for region in os.listdir(folder_path):

        region_path = os.path.join(folder_path, region)

        results_path = os.path.join(result_path, region)
        os.makedirs(results_path, exist_ok=True)

        res = []

        for building in os.listdir(region_path):

            building_id = building.rsplit(".csv",1)[0]

            if building.endswith('.csv'):
                file_path = os.path.join(region_path, building)
                df = pd.read_csv(file_path)
                energy_data = df['energy'].values
                dataset = TimeSeriesDataset(energy_data, backcast_length, forecast_length, stride)
                
                # test phase
                model.eval()
                test_losses = []
                y_true_test = []
                y_pred_test = []

                # test loop
                for x_test, y_test in tqdm(DataLoader(dataset, batch_size=1), desc=f"Testing {building_id}", leave=False):
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    with torch.no_grad():
                        backcast, forecast = model(x_test)
                        loss = criterion(forecast, y_test)
                        test_losses.append(loss.item())
                        
                        # Collect true and predicted values for RMSE calculation
                        y_true_test.extend(y_test.cpu().numpy())
                        y_pred_test.extend(forecast.cpu().numpy())
                        
                # Calculate average validation loss and RMSE
                y_true_combine = np.concatenate(y_true_test, axis=0)
                y_pred_combine = np.concatenate(y_pred_test, axis=0)
                avg_test_loss = np.mean(test_losses)
                
                y_pred_combine_unscaled = unscale_predictions(y_pred_combine, dataset.mean, dataset.std)
                y_true_combine_unscaled = unscale_predictions(y_true_combine, dataset.mean, dataset.std)
                
                # Calculate CVRMSE, NRMSE, MAE on unscaled data
                cvrmse = cal_cvrmse(y_pred_combine_unscaled, y_true_combine_unscaled)
                nrmse = cal_nrmse(y_pred_combine_unscaled, y_true_combine_unscaled)
                mae = cal_mae(y_pred_combine_unscaled, y_true_combine_unscaled)
                mse = cal_mse(y_pred_combine_unscaled, y_true_combine_unscaled)
                mae_norm = cal_mae(y_pred_combine, y_true_combine)
                mse_norm = cal_mse(y_pred_combine, y_true_combine)

                res.append([building_id, cvrmse, nrmse, mae, mae_norm, mse, mse_norm, avg_test_loss])

        columns = ['building_ID', 'CVRMSE', 'NRMSE', 'MAE', 'MAE_NORM', 'MSE', 'MSE_NORM', 'Avg_Test_Loss']
        df = pd.DataFrame(res, columns=columns)
        df.to_csv("{}/{}.csv".format(results_path, 'result'), index=False)



        med_nrmse = df['NRMSE'].median()
        med_mae = df['MAE'].median()
        med_mae_norm = df['MAE_NORM'].median()
        med_mse = df['MSE'].median()
        med_mse_norm = df['MSE_NORM'].median()

        median_res.append([region, med_nrmse, med_mae, med_mae_norm, med_mse, med_mse_norm])

    med_columns = ['Dataset','NRMSE', 'MAE', 'MAE_NORM', 'MSE', 'MSE_NORM']
    median_df = pd.DataFrame(median_res, columns=med_columns)
    median_df.to_csv("{}/{}.csv".format(result_path, 'median_results_of_buildings'), index=False)

                


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    parser.add_argument('--config-file', type=str, default='./configs/energy_data.json', help='Input config file path', required=True)
    parser.add_argument('--seq_len', type=int, default=168, help='Input Sequence Length')
    parser.add_argument('--stride', type=int, default=24, help='Input Stride')
    parser.add_argument('--patch_size', type=int, default=24, help='Input Patch Length')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Input Hidden Dim')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints/MixBEATS', help='Enter model save path')
    parser.add_argument('--result_path', type=str, default='./results/MixBEATS', help="Enter the results path")

    
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
    args['result_path'] = cli_args.result_path


    
    # # Parameters
    backcast_length = args['seq_len']
    forecast_length = args['pred_len']
    stride = args['stride']
    batch_size = args['batch_size']
    patch_size = args['patch_size']
    hidden_dim = args['hidden_dim']
    num_patches = backcast_length // patch_size

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

    model_load_path = '{}/best_model.pth'.format(args['model_save_path'])
    model.load_state_dict(torch.load(model_load_path, weights_only=True))



    # Define loss
    if args['loss'] == 'mse':
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.HuberLoss(reduction="mean", delta=1)


    # training the model and save best parameters
    test(args, model, criterion, device)

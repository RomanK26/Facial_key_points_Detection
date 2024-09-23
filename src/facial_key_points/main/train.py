import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from src.facial_key_points.config.config import configuration
from src.facial_key_points.datasets.datasets import FacialKeyPointsDataset
from src.facial_key_points.model.modified_vgg import get_model
from src.facial_key_points.utils.utils import train, visualization,plot_curve
import json


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA backend")
    else:
        device = torch.device("cpu")
        print("Using CPU backend")


    saved_path = os.path.join(os.getcwd(),'Saved', configuration.get('saved_path'))
    model_path = os.path.join(saved_path,'model.pth')
    hyperparameter_path = os.path.join(saved_path,'hyperparameter.json')
    train_curve_path = os.path.join(saved_path,'train_curve.png')
    # test_curve_path = os.path.join(saved_path,'test_curve.png')

    visualization_path = os.path.join(saved_path,'visualization.png')

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)



    training_data = FacialKeyPointsDataset(csv_file_path=configuration.get('train_data_csv_path'),split = 'training',device = device)
    # print(training_data[0])
    # img_tensor, kp_s = training_data[0]  #-->2
    test_data = FacialKeyPointsDataset(csv_file_path=configuration.get('test_data_csv_path'), split = 'test', device =device)
    train_dataloader = DataLoader(training_data, batch_size=configuration.get('batch_size'), shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=configuration.get('batch_size'), shuffle=False)

    model = get_model(device=device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration.get('learning_rate'))

    train_loss, test_loss = train(configuration['n_epochs'],train_dataloader,test_dataloader, model, criterion, optimizer)
    plot_curve(train_loss, test_loss, train_curve_path,configuration['n_epochs'])
    visualization(img_path='face.jpg',model=model,viz_result_path=visualization_path,model_input_size=configuration['model_input_size'],device=device)


    with open(hyperparameter_path,'w') as h_config:
        json.dump(configuration,h_config)


    torch.save(model,model_path)



if __name__ == '__main__':
    main()
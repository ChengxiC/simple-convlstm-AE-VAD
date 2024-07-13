import torch
from loader import UCSD_test_loader
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from plot import plot_scores


# to compute the distance between the two images
def pixel_error(image1, image2):
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    diff = image1_array - image2_array

    squared = np.square(diff)
    sum_squared = np.sum(squared)

    error = np.sqrt(sum_squared)

    return error


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_data = UCSD_test_loader(700)
loader_test_data = DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=0)
model = torch.load('./pretrained_model/model_best.pt').to(device)

error_max = -np.inf
error_min = np.inf
error_list = []
score_list = []

for batch_idx, x in enumerate(loader_test_data):
    x = x.to(device)
    x_hat = model(x)

    x_flat = x.view(-1).cpu().detach().numpy()
    x_hat_flat = x_hat.view(-1).cpu().detach().numpy()

    error = pixel_error(x_flat, x_hat_flat)

    error_list.append(error.item())

    if error_min > error:
        error_min = error
    if error_max < error:
        error_max = error

for i in range(len(error_list)):
    score = (error_list[i] - error_min) / error_max
    score_list.append(score)

print(error_max)
print(error_min)
plot_scores(700, score_list)







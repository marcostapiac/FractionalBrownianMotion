import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.classes import ClassDenoisingDiffusion, ClassOUDiffusion, ClassSLGDDiffusion
from utils.plotting_functions import plot_loss_epochs


def prepare_data(data: np.ndarray, batch_size: int) -> [DataLoader, DataLoader, DataLoader]:
    S, T = data.shape
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data).float())
    train, val, test = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    trainLoader, valLoader, testLoader = DataLoader(train, batch_size=batch_size, shuffle=True), \
                                         DataLoader(val, batch_size=batch_size, shuffle=True), \
                                         DataLoader(test, batch_size=batch_size, shuffle=True)  # Returns iterator
    return trainLoader, valLoader, testLoader


def train_diffusion_model(diffusion: type[ClassDenoisingDiffusion, ClassSLGDDiffusion, ClassOUDiffusion],
                          trainLoader: torch.utils.data.DataLoader,
                          valLoader: torch.utils.data.DataLoader, opt: torch.optim.Optimizer, nEpochs: int) -> [
    np.array, np.array]:
    # TODO: Need to move variables to correct device -- but which ones?
    train_losses = []
    val_losses = []
    for i in range(nEpochs):
        train_loss = diffusion.one_epoch_diffusion_train(trainLoader=trainLoader, opt=opt)
        val_loss = diffusion.evaluate_diffusion_model(loader=valLoader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            "Percent Completed {:0.4f} Train :: Val Losses, {:0.4f} :: {:0.4f}".format((i + 1) / nEpochs, train_loss,
                                                                                       val_loss))
    return train_losses, val_losses


def save_and_train_diffusion_model(data: np.ndarray, model_filename: str, batch_size: int,
                                   nEpochs: int, lr: float,
                                   diffusion: type[ClassDenoisingDiffusion, ClassSLGDDiffusion, ClassOUDiffusion]) -> \
type[ClassDenoisingDiffusion, ClassSLGDDiffusion, ClassOUDiffusion]:
    """ Prepare data for training """
    trainLoader, valLoader, testLoader = prepare_data(data, batch_size=batch_size)

    """ Prepare optimiser """
    optimiser = torch.optim.Adam((diffusion.parameters()), lr=lr)  # No need to move to device

    """ Training """
    train_loss, val_loss = train_diffusion_model(diffusion=diffusion, trainLoader=trainLoader,
                                                 valLoader=valLoader,
                                                 opt=optimiser, nEpochs=nEpochs)
    plot_loss_epochs(epochs=np.arange(1, nEpochs + 1, step=1), val_loss=val_loss, train_loss=np.array(train_loss))
    print(diffusion.evaluate_diffusion_model(loader=testLoader))
    """ Save model """
    file = open(model_filename, "wb")
    pickle.dump(diffusion, file)
    file.close()
    return diffusion


def check_convergence_at_diffTime(diffusion: type[ClassDenoisingDiffusion, ClassSLGDDiffusion, ClassOUDiffusion],
                                  t: int, dataSamples: np.ndarray) -> [np.ndarray,
                                                                       np.ndarray,
                                                                       list[str]]:
    forward_samples_at_t, _ = diffusion.forward_process(dataSamples=torch.from_numpy(dataSamples),
                                                        diffusionTimes=torch.ones(dataSamples.shape[0],
                                                                                  dtype=torch.long) * (
                                                                           torch.from_numpy(np.array([t]))))
    # Generate backward samples
    backward_samples_at_t = diffusion.reverse_process(dataSize=forward_samples_at_t.shape[0],
                                                      timeDim=forward_samples_at_t.shape[1], timeLim=t + 1)
    labels = ["Forward Samples at time {}".format(t + 1), "Backward Samples at time {}".format(t + 1)]
    return forward_samples_at_t.numpy(), backward_samples_at_t, labels

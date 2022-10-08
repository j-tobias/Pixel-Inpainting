#GENERAL IMPORTS
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#PROJECT IMPORTS
import utils
from architecture import CNN
from datasets import ImageDataset
from datasets import PickleImageDataset
##################################
"""
This file contains the main code it puts together all the pieces designed in the architectue.py, datasets.py and utils.py files
It defines the Hyperparameters, creates instances of the classes and feeds the input, updating steps to the CNN
"""
#HYPERPARAMETERS
with open ('CONFIG.json', mode= 'r') as f:
    hyp_dict = json.load(f)

dataset_path = hyp_dict['dataset_path']
pickle_dataset_path = hyp_dict['pickle_dataset_path']
pickle_or_normal = hyp_dict['pickle_or_normal']
results_path = hyp_dict['results_path']
shuffle = False
bacht_size = hyp_dict['batch_size']
network_config = hyp_dict['network_config']
device = hyp_dict['device']
learningrate = hyp_dict['learningrate']
weight_decay = hyp_dict['weight_decay']
n_updates = hyp_dict['n_updates']
print('1. Hyperparameters initialized')




#set seed
np.random.seed(0)
torch.manual_seed(0)
print('2. Seed set')

# Prepare a path to plot to
plotpath = os.path.join(results_path, "plots")
os.makedirs(plotpath, exist_ok=True)
print('3. Plotting Dir created')

#create Dataset Instance
if pickle_or_normal == 'normal':
    dataset = ImageDataset(dataset_path)
else:
    dataset = PickleImageDataset(pickle_dataset_path)

# Split dataset into training, validation and test set
# is already randomized, so we do not necessarily have to shuffle again)

trainingset = Subset(
    dataset,
    indices=np.arange(int(len(dataset) * (3 / 5)))
)
validationset = Subset(
    dataset,
    indices=np.arange(int(len(dataset) * (3 / 5)), int(len(dataset) * (4 / 5)))
)
testset = Subset(
    dataset,
    indices=np.arange(int(len(dataset) * (4 / 5)), len(dataset))
)
print('4. Datasets set up')

# Store the mean and std to denormalize output.
mean_, std_ = torch.from_numpy(trainingset.mean), torch.from_numpy(trainingset.std)
print('5. Stored Mean and STD')

#Create DataLoaders for the Datasets
trainloader = DataLoader(
    trainingset, 
    bacht_size, 
    shuffle=shuffle, 
    num_workers = 0
)
valloader = DataLoader(
    validationset, 
    bacht_size, 
    shuffle=shuffle, 
    num_workers = 0
)
testloader = DataLoader(
    testset, 
    bacht_size, 
    shuffle=shuffle, 
    num_workers = 0
)
print('6. Loaders set up')

# Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))
print('7. writer set up')

# Create Network
net = CNN(**network_config)
net.to(device)
print('8. Neural Network set up')
# Get adam optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=learningrate, weight_decay=weight_decay)

print_stats_at = 100  # print status to tensorboard every x updates
plot_at = 10_000  # plot every x updates
validate_at = 5000  # evaluate model on validation set and check for new best model every x updates
update = 0  # current update counter
best_validation_loss = np.inf  # best validation loss so far
update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)

# Save initial model as "best" model (will be overwritten later)
saved_model_file = os.path.join(results_path, "best_model.pt")
torch.save(net, saved_model_file)

# Train until n_updates updates have been reached
while update < n_updates:
    for data in trainloader:
        # Get next samples
        #input_array, known_array, target_array, image_array, index
        image, ids, inputs, knowns, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
            
        # Reset gradients
        optimizer.zero_grad()
            
        # Get outputs of our network
        outputs = net(inputs.float())

        _batch_size = outputs.size()[0]     

        # Stack them on themselves N times, where N is the batch size.
        _mean = torch.reshape(mean_, (3, 1))
        _mean = _mean.repeat(_batch_size, 100 * 100)
        _mean = torch.reshape(_mean, (_batch_size, 3, 100, 100))
        _mean = _mean.to(device)

        _std = torch.reshape(std_, (3, 1))
        _std = _std.repeat(_batch_size, 100 * 100)
        _std = torch.reshape(_std, (_batch_size, 3, 100, 100))
        _std = _std.to(device)  

        # Denormalize output
        outputs = utils.denormalize_image(outputs, _mean, _std) 
            
        # Calculate loss, do backward pass and update weights
        loss = utils.mse(outputs, targets.float())
        loss.backward()
        optimizer.step()
            
        # Print current status and score
        if (update + 1) % print_stats_at == 0:
            writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=update)
            
        # Plot output
        if (update + 1) % plot_at == 0:
            utils.plot(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), outputs.detach().cpu().numpy(), plotpath, update)
            
        # Evaluate model on validation set
        if (update + 1) % validate_at == 0:
            val_loss = utils.evaluate_model(net, dataloader=valloader, loss_fn=utils.mse, device=device)
            writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=update)
            # Add weights and gradients as arrays to tensorboard
            for i, (name, param) in enumerate(net.named_parameters()):
                writer.add_histogram(tag=f"validation/param_{i} ({name})", values=param.cpu(), global_step=update)
                writer.add_histogram(tag=f"validation/gradients_{i} ({name})", values=param.grad.cpu(), global_step=update)
            # Save best model for early stopping
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                torch.save(net, saved_model_file)
            
        update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
        update_progress_bar.update()
            
        # Increment update counter, exit if maximum number of updates is reached
        # Here, we could apply some early stopping heuristic and also exit if its
        # stopping criterion is met
        update += 1
        if update >= n_updates:
            break

    update_progress_bar.close()
    writer.close()
    print("Finished Training!")
    
    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(saved_model_file)
    train_loss = utils.evaluate_model(net, dataloader=trainloader, loss_fn=utils.mse, device=device)
    val_loss = utils.evaluate_model(net, dataloader=valloader, loss_fn=utils.mse, device=device)
    test_loss = utils.evaluate_model(net, dataloader=testloader, loss_fn=utils.mse, device=device)
    
    print(f"Scores:")
    print(f"  training loss: {train_loss}")
    print(f"validation loss: {val_loss}")
    print(f"      test loss: {test_loss}")
    
    # Write result to file
    with open(os.path.join(results_path, "results.txt"), "w") as rf:
        print(f"Scores:", file=rf)
        print(f"  training loss: {train_loss}", file=rf)
        print(f"validation loss: {val_loss}", file=rf)
        print(f"      test loss: {test_loss}", file=rf)

        
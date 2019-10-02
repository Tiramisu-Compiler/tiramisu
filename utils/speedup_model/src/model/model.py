import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)

e = 0.05

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        
        hidden2_size = int(input_size / 2)
        hidden1_size = int((input_size + hidden2_size) * 3 / 2)
        hidden3_size = int((output_size + hidden2_size) * 3 / 2)
        
        self.hidden1 = nn.Linear(input_size, hidden1_size)
        self.hidden2 = nn.Linear(hidden1_size, hidden2_size)
        self.hidden3 = nn.Linear(hidden2_size, hidden3_size)

        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.xavier_uniform_(self.hidden2.weight)
        nn.init.xavier_uniform_(self.hidden3.weight)

        self.predict = nn.Linear(hidden3_size, output_size)
        nn.init.xavier_uniform_(self.predict.weight)
        
        

    def forward(self, x):
        
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        
        x = self.predict(x)
        #x = F.relu(self.predict(x))
        
        return x
    
    
def train_model(model, criterion, optimizer, dataloader, num_epochs=100):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 5000.0
    losses = []
    train_loss = 0
    model = model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train()  
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            
            # Iterate over data.
            for inputs, labels in tqdm(dataloader[phase], total=len(dataloader[phase])):       
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  
                    assert outputs.shape == labels.shape

                    loss = torch.sqrt(criterion(outputs, labels))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                       # print(loss.item())
                        
            
                # statistics
                running_loss += loss.item()     
                
                #running_corrects += torch.sum((outputs.data - labels.data) < e)/inputs.shape[0]

            epoch_loss = running_loss / len(dataloader[phase])
            epoch_acc = epoch_loss #running_corrects.double() /len(dataloader)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
               phase, epoch_loss, epoch_acc))

            #deep copy the model
            if phase == 'val' and epoch_acc < best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            
            if phase == 'val':
                losses.append((train_loss, epoch_loss))
            else:
                train_loss = epoch_loss

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, losses

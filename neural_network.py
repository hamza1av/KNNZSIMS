# we take the results of the simulation and adjust the state vector to an input and time shifted output vector

# the simulation needs to be done ...

X = x_out[:-1]
Y = x_out[1:]
print(np.shape(X))
print(np.shape(Y))

print(X[-4:-1])
print(Y[-4:-1])


# numpy array manipulation
number_states = 4
k = 15   #number of timesteps we take into account to predict the future.

n = number_samples-k

new_x = np.zeros((n,number_states*k))

for i in range(n):
    new_x[i,:] = x_out[i:i+k].flatten()

new_y = x_out[k:]
########## das machen wir damit die letzten k zustaende benutzt werden um die vorhersage fuer den naechsten zustand zu treffen

## Nun konvertieren die numpy arrays in pytorch tensoren
input_tensor = torch.from_numpy(new_x).float()
target_tensor = torch.from_numpy(new_y).float()

input_tensor = input_tensor.unsqueeze(1)  # Adding a dimension to represent sequence length
target_tensor = target_tensor.unsqueeze(1)  # Adding a dimension to represent sequence length

##########3

############# Hier kommt das eigentliche Neuronale Netzt

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Parameters for the RNN cell
        self.Wxh = nn.Linear(input_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        combined = self.Wxh(x) + self.Whh(h)
        return torch.tanh(combined)


class MultiLayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MultiLayerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define a list of custom RNN cells
        self.rnn_cells = nn.ModuleList([CustomRNNCell(input_size if i == 0 else hidden_size, hidden_size)
                                        for i in range(num_layers)])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #batch_size = x.size(0) #alt, war falsch, wieso auch immer, hat den fehler verursacht mit den hohen dimensionalitaeten
        batch_size = 1
        #print(f'batch size = {batch_size}')    #zum debuggen benutzt

        # Initialize hidden state with zeros for each layer
        hidden_states = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]
    
        # Forward through multiple RNN layers
        for layer_idx in range(self.num_layers):
            # Get the current RNN cell
            rnn_cell = self.rnn_cells[layer_idx]

            # Forward through the RNN cell
            hidden_states[layer_idx] = rnn_cell(x if layer_idx == 0 else hidden_states[layer_idx - 1],
                                                hidden_states[layer_idx])

        # Take the output from the last layer
        out = self.fc(hidden_states[-1])
        return out


input_dim = number_states * k
output_dim = number_states
hidden_dim = 8
num_layers = 8

model = MultiLayerRNN(input_dim, hidden_dim, output_dim, num_layers)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

#Hier ist die Trainingsphase
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Forward pass
    outputs = model(input_tensor)
    loss = criterion(outputs, target_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


### Bis hierhin waere das Neuroanlale Netzt trainiert. 
import torch
import numpy as np
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import PostPre
from tqdm import tqdm

# Basic parameters
time_bins = 100
input_neurons = 224 * 224
hidden_neurons = 100
output_neurons = 2
batch_size = 32
learning_rate = 1e-3
num_epochs = 10

# Load spike train data and labels
train_spikes = np.load("train_spikes.npy")
test_spikes = np.load("test_spikes.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_spikes = torch.tensor(train_spikes, dtype=torch.float32).to(device)
test_spikes = torch.tensor(test_spikes, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

network = Network()

# Enable spike traces for STDP
input_layer = Input(n=input_neurons, shape=(1, input_neurons), traces=True)
network.add_layer(input_layer, name="Input")

hidden_layer = LIFNodes(n=hidden_neurons, traces=True)
network.add_layer(hidden_layer, name="Hidden")

output_layer = LIFNodes(n=output_neurons, refrac=0)
network.add_layer(output_layer, name="Output")

input_hidden_conn = Connection(source=input_layer, target=hidden_layer,
                               w=torch.rand(input_neurons, hidden_neurons) * 0.1)
network.add_connection(input_hidden_conn, source="Input", target="Hidden")

hidden_output_conn = Connection(source=hidden_layer, target=output_layer,
                                w=torch.rand(hidden_neurons, output_neurons) * 0.1)
network.add_connection(hidden_output_conn, source="Hidden", target="Output")

input_hidden_conn.update_rule = PostPre(connection=input_hidden_conn, nu=learning_rate)

def train():
    network.train(mode=True)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        permutation = torch.randperm(train_spikes.size(0))
        correct = 0
        for i in tqdm(range(0, train_spikes.size(0), batch_size)):
            idx = permutation[i:i + batch_size]
            inputs = train_spikes[idx].transpose(1, 2)
            labels = y_train[idx]

            network.reset_state_variables()

            for step in range(time_bins):
                input_t = {"Input": inputs[:, :, step]}
                network.run(inputs=input_t, time=1)

            # Sum spikes over time dimension for output layer
            spikes_output = network.layers["Output"].spike_record.sum(dim=1)
            predicted = spikes_output.argmax(dim=1)
            correct += (predicted == labels).sum().item()

            network.reset_state_variables()
        accuracy = correct / train_spikes.size(0)
        print(f"Training accuracy: {accuracy:.4f}")

def test():
    network.train(mode=False)
    correct = 0
    for i in range(0, test_spikes.size(0), batch_size):
        inputs = test_spikes[i:i + batch_size].transpose(1, 2)
        labels = y_test[i:i + batch_size]

        network.reset_state_variables()

        for step in range(time_bins):
            input_t = {"Input": inputs[:, :, step]}
            network.run(inputs=input_t, time=1)

        spikes_output = network.layers["Output"].spike_record.sum(dim=1)
        predicted = spikes_output.argmax(dim=1)
        correct += (predicted == labels).sum().item()

        network.reset_state_variables()

    accuracy = correct / test_spikes.size(0)
    print(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train()
    test()

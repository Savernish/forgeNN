import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- 1. Define the Predictive Coding Network ---
class PredictiveCodingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define layers. These are the weights of the generative model.
        self.fc2 = nn.Linear(output_size, hidden_size, bias=False) # Top-down
        self.fc1 = nn.Linear(hidden_size, input_size, bias=False)  # Hidden to input
        
        # Activation function
        self.activation = nn.Tanh()

    def initialize_states(self, batch_size, device):
        # Initialize neuron values (x) for each layer
        # Note: nn.Linear(in_features, out_features)
        # self.fc1 maps hidden -> input, so fc1.in_features == hidden_size, fc1.out_features == input_size
        # self.fc2 maps output -> hidden, so fc2.in_features == output_size, fc2.out_features == hidden_size
        x0 = torch.zeros(batch_size, self.fc1.out_features).to(device)  # Input layer (batch, input_size)
        x1 = torch.zeros(batch_size, self.fc1.in_features).to(device)   # Hidden layer (batch, hidden_size)
        x2 = torch.zeros(batch_size, self.fc2.in_features).to(device)   # Output layer (batch, output_size)
        return x0, x1, x2

    def forward(self, input_data, inference_steps=20, lr_inference=0.1):
        """
        The forward pass performs inference by updating neuron values (x)
        to minimize prediction error. It does NOT use backpropagation.
        """
        batch_size = input_data.shape[0]
        device = input_data.device

        # Initialize neuron values for this batch
        x0, x1, x2 = self.initialize_states(batch_size, device)
        x0 = input_data # Clamp the input layer to the data

        # --- Inference Phase ---
        # Iteratively update neuron values to minimize prediction error
        for i in range(inference_steps):
            # Calculate predictions (top-down)
            pred1 = self.activation(self.fc2(x2)) # Prediction of hidden layer
            pred0 = self.activation(self.fc1(x1)) # Prediction of input layer

            # Calculate prediction errors (bottom-up)
            e1 = x1 - pred1
            e0 = x0 - pred0
            
            # Update neuron values via gradient descent on the error
            # Use batch-compatible matrix multiplies. Shapes:
            # e1: (batch, hidden_size)
            # self.fc2.weight: (hidden_size, output_size) because fc2 is Linear(output_size, hidden_size)
            # To get gradient w.r.t. x2 (batch, output_size) we compute e1 @ fc2.weight -> (batch, output_size)
            grad_x2 = e1 @ self.fc2.weight  # (batch, output_size)
            x2 += lr_inference * grad_x2

            # For x1 update: self.fc1.weight has shape (input_size, ???) but fc1 is Linear(hidden_size, input_size)
            # self.fc1.weight: (input_size, hidden_size)
            # e0: (batch, input_size)
            # e0 @ self.fc1.weight -> (batch, hidden_size)
            grad_x1 = e0 @ self.fc1.weight - e1
            x1 += lr_inference * grad_x1

        return x0, x1, x2

# --- 2. Training and Data Loading ---
def train_pc_network(model, data_loader, epochs=5, lr_learning=0.001, device='cpu'):
    model.to(device)
    # Use a standard optimizer for the learning phase (weight updates)
    optimizer = optim.SGD(model.parameters(), lr=lr_learning)

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.view(data.shape[0], -1).to(device) # Flatten images
            
            # --- Learning Phase ---
            # 1. Run inference to get the settled neuron states
            with torch.no_grad(): # Don't track gradients during inference
                x0, x1, x2 = model(data)

            # 2. Update weights based on the settled states
            # This is where we update the model's parameters
            optimizer.zero_grad()
            
            # The "loss" for PC is the sum of squared prediction errors
            pred1 = model.activation(model.fc2(x2))
            pred0 = model.activation(model.fc1(x1))
            
            loss_e1 = 0.5 * (x1 - pred1).pow(2).sum()
            loss_e0 = 0.5 * (x0 - pred0).pow(2).sum()
            loss = loss_e0 + loss_e1
            
            # Backpropagate the error to update weights
            # This is a convenient way to implement the PC weight update rule,
            # which is equivalent to gradient descent on the prediction error.
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(data_loader)} | Loss: {loss.item():.4f}")
        
        print(f"--- Epoch {epoch+1} Finished | Average Loss: {total_loss/len(data_loader):.4f} ---")

# --- 3. Main Execution ---
if __name__ == '__main__':
    # Hyperparameters
    BATCH_SIZE = 64
    INPUT_SIZE = 784  # 28*28 pixels
    HIDDEN_SIZE = 256
    OUTPUT_SIZE = 64  # Size of the learned representation
    EPOCHS = 3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # Load MNIST Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize and Train the PC Network
    pc_model = PredictiveCodingNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    train_pc_network(pc_model, train_loader, epochs=EPOCHS, device=DEVICE)

    # --- 4. Evaluate the Learned Representations ---
    # After unsupervised training, we test how good the features are.
    # We train a simple linear classifier on top of the frozen PC features.
    
    print("\n--- Evaluating representations ---")
    classifier = nn.Linear(OUTPUT_SIZE, 10).to(DEVICE) # 10 classes for MNIST
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    pc_model.eval() # Put PC model in evaluation mode
    for epoch in range(2): # Train classifier for a few epochs
        for data, target in train_loader:
            data = data.view(data.shape[0], -1).to(DEVICE)
            target = target.to(DEVICE)
            
            # Get representations from the trained PC network
            with torch.no_grad():
                _, _, representation = pc_model(data)

            # Train the classifier
            classifier_optimizer.zero_grad()
            output = classifier(representation)
            loss = criterion(output, target)
            loss.backward()
            classifier_optimizer.step()
    
    # Final accuracy test
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in train_loader: # Using train loader for simplicity
            data = data.view(data.shape[0], -1).to(DEVICE)
            target = target.to(DEVICE)
            _, _, representation = pc_model(data)
            output = classifier(representation)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"\nFinal Accuracy on Training Set: {100 * correct / total:.2f}%")
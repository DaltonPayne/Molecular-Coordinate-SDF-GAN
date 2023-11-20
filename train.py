import torch.nn as nn
from rdkit import Chem
import torch
import os

# Define the Generator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Hyperparameters
input_dim = 100  # This can be any arbitrary number. It's the size of the random input to the generator.
output_dim = 3   # Let's say we are generating coordinates for one atom at a time.

G = Generator(input_dim, output_dim)
D = Discriminator(output_dim)

# Loss
criterion = nn.BCELoss()

# Optimizers
lr = 0.0002
g_optimizer = torch.optim.Adam(G.parameters(), lr=lr)
d_optimizer = torch.optim.Adam(D.parameters(), lr=lr)

# Hyperparameters
num_epochs = 1000
batch_size = 64  # Assuming you have enough data or you might need to adjust this
real_label = 1
fake_label = 0

# Sample noise to see how the generator improves over time
fixed_noise = torch.randn(64, input_dim)

for epoch in range(num_epochs):

    for i, data in enumerate(sdf_files, 0):  # Assuming each sdf_file is a batch of tensors
        # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        D.zero_grad()

        # Train with real data
        real_data = sdf_to_tensor(os.path.join(sdf_directory, data))
        label = torch.full((batch_size,), real_label)
        output = D(real_data).view(-1)
        d_loss_real = criterion(output, label)
        d_loss_real.backward()

        # Train with fake data
        noise = torch.randn(batch_size, input_dim)
        fake_data = G(noise)
        label.fill_(fake_label)
        output = D(fake_data.detach()).view(-1)
        d_loss_fake = criterion(output, label)
        d_loss_fake.backward()

        d_optimizer.step()

        # (2) Update Generator: maximize log(D(G(z)))
        G.zero_grad()
        label.fill_(real_label)  # Generator wants to make the discriminator think the data is real
        output = D(fake_data).view(-1)
        g_loss = criterion(output, label)
        g_loss.backward()

        g_optimizer.step()

    # Print losses for the epoch
    print(f"[{epoch}/{num_epochs}] D Loss: {d_loss_real.item() + d_loss_fake.item()} | G Loss: {g_loss.item()}")

    # Save real and fake data to visualize progress
    if epoch % 100 == 0:
        with torch.no_grad():
            fake = G(fixed_noise)
            # Here you can save or visualize the generated tensors to see the progress


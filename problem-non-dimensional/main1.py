import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Define the neural network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

# Define the physics-informed loss function
def physics_informed_loss(model, x, y, z, t, rho, Cp, k, q_laser, h_c, T_a, sigma, epsilon, T_0):
    T = model(torch.cat([x, y, z, t], dim=1))

    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_z = torch.autograd.grad(T, z, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_t = torch.autograd.grad(T, t, grad_outputs=torch.ones_like(T), create_graph=True)[0]

    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y), create_graph=True)[0]
    T_zz = torch.autograd.grad(T_z, z, grad_outputs=torch.ones_like(T_z), create_graph=True)[0]

    # PDE residual
    residual_pde = rho * Cp * T_t - (k * (T_xx + T_yy + T_zz)) - q_laser

    # Initial condition residual
    residual_init = T - T_0

    # Boundary condition residual
    q_c = h_c * (T - T_a)
    q_r = sigma * epsilon * (T**4 - T_a**4)
    residual_bc = k * (T_x + T_y + T_z) + q_c + q_r

    loss = torch.mean(residual_pde**2) + torch.mean(residual_init**2) + torch.mean(residual_bc**2)
    return loss

# Hyperparameters
learning_rate = 0.001
num_epochs = 10000

# Define the domain for x, y, z, and t
num_points = 1000
x = torch.rand(num_points, 1, requires_grad=True)
y = torch.rand(num_points, 1, requires_grad=True)
z = torch.rand(num_points, 1, requires_grad=True)
t = torch.rand(num_points, 1, requires_grad=True)

# Physical parameters (replace with actual values)
rho = 7850  # Density (kg/m^3)
Cp = 500    # Specific heat (J/(kg*K))
k = 45      # Thermal conductivity (W/(m*K))
q_laser = 1000  # Laser heat source term (W/m^3)
h_c = 10    # Convective heat transfer coefficient (W/(m^2*K))
T_a = 300   # Ambient temperature (K)
sigma = 5.67e-8  # Stefan-Boltzmann constant (W/(m^2*K^4))
epsilon = 0.9  # Emissivity
T_0 = 300   # Initial temperature (K)

# Convert parameters to tensors
rho = torch.tensor(rho)
Cp = torch.tensor(Cp)
k = torch.tensor(k)
q_laser = torch.tensor(q_laser)
h_c = torch.tensor(h_c)
T_a = torch.tensor(T_a)
sigma = torch.tensor(sigma)
epsilon = torch.tensor(epsilon)
T_0 = torch.tensor(T_0)

# Create the model
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = physics_informed_loss(model, x, y, z, t, rho, Cp, k, q_laser, h_c, T_a, sigma, epsilon, T_0)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'pinn_model.pth')

# Plotting the results
def plot_temperature_field(model, x, y, z, t):
    with torch.no_grad():
        input_data = torch.cat([x, y, z, t], dim=1)
        T_pred = model(input_data).cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy(), c=T_pred, cmap='viridis')
    fig.colorbar(scatter, ax=ax, label='Temperature')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    plt.savefig('temperature_field.png')


# Generate a grid of points for plotting
x_plot = torch.linspace(0, 1, 50).reshape(-1, 1)
y_plot = torch.linspace(0, 1, 50).reshape(-1, 1)
z_plot = torch.linspace(0, 1, 50).reshape(-1, 1)
t_plot = torch.zeros_like(x_plot)

plot_temperature_field(model, x_plot, y_plot, z_plot, t_plot)

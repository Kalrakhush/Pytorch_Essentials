import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

lr=0.1
model=nn.Linear(10,1)

optimizer=torch.optim.Adam(model.parameters(),lr=lr)

lambda1=lambda epoch: epoch/10
scheduler=lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1) # Custom learning rate scheduler

mul_scheduler= lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.95) # Multiplicative learning rate scheduler

step_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # Step learning rate scheduler

print("Initial learning rate:", optimizer.param_groups[0]['lr'])
for epoch in range(20):
    optimizer.step()
    step_scheduler.step()
    print(f"Epoch {epoch+1}: Learning rate: {optimizer.param_groups[0]['lr']}")
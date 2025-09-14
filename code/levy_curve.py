import torch
import matplotlib.pyplot as plt

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def levy_c_curve(start, end, depth):
    """
    Generate Levy C curve points recursively.
    
    start, end: tensors of shape (2,) representing points
    depth: recursion depth
    """
    if depth == 0:
        return torch.stack([start, end], dim=0)
    
    # midpoint rotated by 45 degrees
    mid = (start + end) / 2
    
    # vector from start to end
    vec = end - start
    
    # rotate by 45 degrees CCW and scale by 1/sqrt(2)
    rotation = torch.tensor([[0.5, -0.5], [0.5, 0.5]], device=device)
    new_vec = vec @ rotation.T
    mid = start + new_vec
    
    # recursively generate two halves
    left = levy_c_curve(start, mid, depth-1)
    right = levy_c_curve(mid, end, depth-1)
    
    # combine, skip the midpoint duplication
    return torch.cat([left[:-1], right], dim=0)

# initial line
p0 = torch.tensor([0.0, 0.0], device=device)
p1 = torch.tensor([1.0, 0.0], device=device)

# recursion depth
depth = 20

points = levy_c_curve(p0, p1, depth).cpu().numpy()

# plot
plt.figure(figsize=(8, 8))
plt.plot(points[:,0], points[:,1], color='blue')
plt.axis('equal')
plt.axis('off')
plt.show()

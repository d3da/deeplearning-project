### File: 4090_RES/ENSEMBLE

#### Testing Parameters:
- **STEP_SIZE**: 9e-3  
- **NUM_STEPS**: 10  

---

### Ensemble Results:
- **Normal Accuracy**: 99.92%  
- **Adversarial Accuracy**: 38.64%  

---

### Results for REGNET:
- **Normal Accuracy**: 99.70%  
- **Adversarial Accuracy**: 24.88%  

---

### Results for MAXVIT:
- **Normal Accuracy**: 98.72%  
- **Adversarial Accuracy**: 49.99%  

---

### Results for EFFICIENTNET:
- **Normal Accuracy**: 98.86%  
- **Adversarial Accuracy**: 3.92%  

--- 



class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        # returns <10% after 1 step without defense
        self.defense = DefenseTransform(kernel_size=3, noise_std=0.1)

    def forward(self, x):
        x = self.defense(x)
        logits = [model(x) for model in self.models]
        return torch.mean(torch.stack(logits), dim=0)

class DefenseTransform(nn.Module):
    def __init__(self, kernel_size=3, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std
        self.gaussian_blur = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
        
    def forward(self, x):
        # adding in gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # applying gaussian blur 
        x = self.gaussian_blur(x)
        
        # Clip values to maintain valid range
        x = torch.clamp(x, -2.5, 2.5)  # Adjusted for normalized inputs
        return x



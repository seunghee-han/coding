def add_noise(img):
    device='cuda'
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise.to(device)
  
    # if random.random() < 0.5:
    #     noisy_img = transforms.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)(noisy_img)

    return noisy_img

def apply_random_augmentation(img):
    # 각 증강 기법을 독립적으로 적용
    if random.random() < 0.5:
        img = transforms.RandomHorizontalFlip(p=1.0)(img)
    
    if random.random() < 0.5:
        img = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))(img)
    
    if random.random() < 0.5:
        img = transforms.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)(img)
    
    return img

def random_img(img,output_dir='augmented_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    augmented_img = torch.stack([apply_random_augmentation(img[i]) for i in range(img.size(0))])
    _, _, height, _ = augmented_img.size()
    half_height = height // 2
    upper_height = height // 4
    lower_height = 3 * height // 4
    # 상체, 중간, 하체를 마스킹할 텐서 생성
    upper_body_masked = torch.zeros_like(augmented_img)
    middle_body_masked = torch.zeros_like(augmented_img)
    lower_body_masked = torch.zeros_like(augmented_img)
    
    for idx in range(img.size(0)):
        # 상체부분을 마스킹하지 않고, 하체부분을 0
        upper_body_masked[idx, :, :half_height, :] = augmented_img[idx, :, :half_height, :]
        middle_body_masked[idx, :, upper_height:lower_height, :] = augmented_img[idx, :, upper_height:lower_height, :]
        # 하체부분을 마스킹하지 않고, 상체부분을 0
        lower_body_masked[idx, :, half_height:, :] = augmented_img[idx, :, half_height:, :]


        utils.save_image(augmented_img[idx], os.path.join(output_dir, f'augmented_{idx}.png'))

    return upper_body_masked, middle_body_masked, lower_body_masked

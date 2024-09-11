def apply_random_augmentation(img):
    augmentations = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
    ]
    
    # 무작위로 증강 기법 선택
    aug = random.choice(augmentations)
    return aug(img)

def random_img(img):
    augmented_img = torch.stack([apply_random_augmentation(img[i]) for i in range(img.size(0))])
    _, _, height, _ = augmented_img.size()
    half_height = height // 2
    upper_height = height // 4
    lower_height = 3 * height // 4
    # 상체를 마스킹할 텐서와 하체를 마스킹할 텐서 생성
    upper_body_masked = torch.zeros_like(augmented_img)
    lower_body_masked = torch.zeros_like(augmented_img)
    middle_body_masked = torch.zeros_like(augmented_img)

    # 이미지의 각 행에 대해 상체 부분과 하체 부분 각각에 업데이트
    for idx in range(img.size(0)):
        # 상체부분을 마스킹하지 않고, 하체부분을 0
        upper_body_masked[idx, :, :half_height, :] = augmented_img[idx, :, :half_height, :]
        middle_body_masked[idx, :, upper_height:lower_height, :] = augmented_img[idx, :, upper_height:lower_height, :]
        # 하체부분을 마스킹하지 않고, 상체부분을 0
        lower_body_masked[idx, :, half_height:, :] = augmented_img[idx, :, half_height:, :]

    return upper_body_masked, middle_body_masked, lower_body_masked

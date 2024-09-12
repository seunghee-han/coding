def mask_images(img):
    _, _, height, _ = img.size()
    half_height = height // 2
    upper_height = height // 4
    lower_height = 3 * height // 4
    # 상체,중간,하체를 마스킹할 텐서 생성
    upper_body_masked = torch.zeros_like(img)
    lower_body_masked = torch.zeros_like(img)
    middle_body_masked = torch.zeros_like(img)

    # 이미지의 각 행에 대해 상체 부분과 하체 부분 각각에 업데이트
    for idx in range(img.size(0)):
        # 하체부분을 0
        upper_body_masked[idx, :, :half_height, :] = img[idx, :, :half_height, :]
        # 중간부분빼고 다 0
        middle_body_masked[idx, :, upper_height:lower_height, :] = img[idx, :, upper_height:lower_height, :]
        # 상체부분을 0
        lower_body_masked[idx, :, half_height:, :] = img[idx, :, half_height:, :]
    return upper_body_masked, middle_body_masked, lower_body_masked

import cv2
import os
import natsort

## 사용법
# 라벨 넘버 설정후 구동 하면 댐
# c1 ~ c11 까지 자동으로 불러와지기 때문에 정해진 라벨만 설정 후 돌리면 파일명까지 c 별로 나눠져서 나옴
# ROI 영역 그려준 후, enter키를 누르면 다음 이미지로 넘어가고 파일 저장이 된다.

label = "01"  # 01 ~ 11
# 이미지가 저장된 폴더 경로 및 결과 저장 폴더 경로
image_folder = '/data/boundingbox_crop/'  # 잘라낼 이미지가 있을 폴더 경로를 입력하세요
output_folder = '/data/crop_image'  # 잘라낸 이미지를 저장할 폴더 경로를 입력하세요



# 결과 저장 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# 폴더 내의 모든 이미지 파일에 대해 반복
for f in natsort.natsorted(os.listdir(image_folder)):
    cnt = 0
    for filename in natsort.natsorted(os.listdir(image_folder + "/" + f)):

        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):  # 확장자 확인
            # 이미지 파일 경로 설정
            img_path = os.path.join(image_folder + "/" + f , filename)

            # 이미지 읽기
            img = cv2.imread(img_path)

            if img is None:
                print(f"이미지를 불러오는데 실패했습니다: {filename}")
                continue

            # 사용자가 ROI 영역 선택 (cv2.selectROI는 마우스 입력을 기다림)
            roi = cv2.selectROI(img)

            # ROI 영역에서 잘라내기
            if roi is not None and roi != (0, 0, 0, 0):
                x, y, w, h = roi
                crop_img = img[int(y):int(y+h), int(x):int(x+w)]

                # 잘라낸 이미지 저장
                # 결과 저장 폴더가 없으면 생성
                if not os.path.exists(output_folder + "/" + f):
                    os.makedirs(output_folder + "/" + f)
                cv2.imwrite(output_folder + "/" + f + "/c{}_{}_{}.jpg".format(str(int(f)), label, cnt), crop_img)
                cnt = cnt+1

                # print(f"{filename} 잘라내기 완료 -> {output_path}")
            else:
                print(f"{filename}에서 잘라낸 영역이 없습니다")

            # ROI 선택 창을 닫음
            cv2.destroyWindow("이미지를 클릭하고 드래그하여 ROI 영역을 선택하세요")

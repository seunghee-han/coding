import cv2
import numpy as np

# 전역 변수 설정
drawing = False  # 마우스가 클릭된 상태 확인
ix, iy = -1, -1  # 시작점 좌표
image = None  # 원본 이미지
image_copy = None  # 작업용 이미지 복사본

# 마우스 이벤트 콜백 함수
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, image, image_copy

    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 마우스 버튼 클릭
        drawing = True
        ix, iy = x, y
        image_copy = image.copy()

    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 이동
        if drawing:
            image_copy = image.copy()
            cv2.rectangle(image_copy, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:  # 왼쪽 마우스 버튼 놓음
        drawing = False
        cv2.rectangle(image_copy, (ix, iy), (x, y), (0, 255, 0), 2)

        # 바운딩 박스 영역 추출 (선 제외)
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        roi = image[y1:y2, x1:x2]

        # 추출된 영역 저장
        output_path = f"/workspace/cau_dataset/img/c1_01_{x1}.jpg"
        cv2.imwrite(output_path, roi)
        print(f"바운딩 박스 이미지가 {output_path}에 저장되었습니다.")
        print(f"바운딩 박스 좌표: (x={x1}, y={y1}, width={x2-x1}, height={y2-y1})")

# 메인 함수
def main():
    global image, image_copy

    # 이미지 로드
    image_path = ""  # 이미지 경로
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 불러올 수 없습니다.")
        return

    image_copy = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_rectangle)

    while True:
        cv2.imshow("image", image_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 'q' 키를 누르면 종료
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

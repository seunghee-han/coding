import os
import glob

def delete_cam_files(folder_path):
    # 지정된 폴더 경로의 모든 파일 목록 가져옴
    all_files = glob.glob(os.path.join(folder_path, '*'))
    deleted_count = 0
    
    for file in all_files:
        # 파일 이름에 '????'이 포함되어 있는지 확인
        if '????' in os.path.basename(file).lower():
            try:
                # 파일 삭제
                os.remove(file)
                print(f"삭제됨: {file}")
                # 삭제된 파일 수
                deleted_count += 1
            except Exception as e:
                print(f"파일 삭제 중 오류 발생: {file}")
                print(f"오류 메시지: {str(e)}")
    
    print(f"!! 총 {deleted_count}개의 파일 삭제 !!")

# 폴더 경로 지정
folder_path = ''

# 실행
delete_cam_files(folder_path)

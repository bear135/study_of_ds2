import pandas as pd
import json
import requests
import uuid
import time
import os

# NCP OCR API 설정
secret_key = 'Z3Z0Tmp6RlVBUXJ2aWtCTXFKVkhVT1l4cEpnanBBSmM='
api_url = 'https://aoomy83n4j.apigw.ntruss.com/custom/v1/40587/fd836bf8392357725bcaf84a6449ed0c64e482a5e27c13ac48781984cec6976e/infer'

# 스캔 이미지가 저장된 폴더 지정
scan_folder = 'scan/'

# 각 파일의 텍스트 데이터를 저장할 딕셔너리
extracted_data = {}

# 폴더 내 모든 이미지 처리
for img_file in os.listdir(scan_folder):
    if img_file.endswith('.png'):
        img_path = os.path.join(scan_folder, img_file)
        filename = os.path.splitext(img_file)[0]  # 확장자 제거하여 파일명 추출
        
        # OCR API 호출
        files = [('file', open(img_path, 'rb'))]
        request_json = {
            'images': [
                {
                    'format': 'png',
                    'name': img_file,
                    'templateIds': [36898]
                }
            ],
            'requestId': str(uuid.uuid4()),
            'version': 'V2',
            'timestamp': int(round(time.time() * 1000))
        }
        
        payload = {'message': json.dumps(request_json).encode('UTF-8')}
        headers = {'X-OCR-SECRET': secret_key}

        response = requests.request("POST", api_url, headers=headers, data=payload, files=files)
        result = response.json()

        # 각 파일별 텍스트 저장
        extracted_data[filename] = [field['inferText'] for field in result['images'][0]['fields']]

# 최대 길이 설정하여 DataFrame 변환
max_length = max(len(texts) for texts in extracted_data.values())
for key in extracted_data:
    extracted_data[key] += [''] * (max_length - len(extracted_data[key]))  # 길이 맞추기

df = pd.DataFrame(extracted_data)
df.index.name = "Index"

# CSV 파일로 저장 (인덱스 유지)
df.to_csv('output/extracted_texts.csv', index=True, encoding='utf-8-sig')

print("텍스트 추출 완료! output/extracted_texts.csv 파일을 확인하세요.")
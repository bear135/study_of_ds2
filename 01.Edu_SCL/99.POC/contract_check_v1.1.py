import os
import sys
import json
import time
import argparse
import pandas as pd
from tkinter import Tk, filedialog
from docx import Document
import pdfplumber
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from datetime import datetime

# -----------------------------
# 환경 설정
# -----------------------------
load_dotenv()
API_KEY = "*****"
MODEL_NAME = "gpt-4o-mini"
MAX_THREADS = 2

# -----------------------------
# 파일 선택
# -----------------------------
def select_contract_file():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="검토할 문서 선택",
        filetypes=[("Word or PDF Documents", "*.docx *.pdf")]
    )

def select_prompt_file():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return filedialog.askopenfilename(
        title="프롬프트 파일 선택",
        filetypes=[("Text Files", "*.txt")]
    )

def read_prompt():
    prompt_path = select_prompt_file()
    if not prompt_path or not os.path.exists(prompt_path):
        print("⚠️ 프롬프트 파일이 선택되지 않았거나 존재하지 않습니다.")
        return "다음 문장을 검토하세요."
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

# -----------------------------
# 파일 읽기
# -----------------------------
def read_docx(file_path):
    doc = Document(file_path)
    return [{"text": p.text.strip(), "page": 1} for p in doc.paragraphs if p.text.strip()]

def read_pdf(file_path):
    items_with_page = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                lines = [line.strip() for line in page_text.split("\n") if line.strip()]
                for line in lines:
                    items_with_page.append({"text": line, "page": page_num})
    return items_with_page

# -----------------------------
# JSON 안전 파싱
# -----------------------------
def safe_json_loads(text: str):
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [parsed]
        elif isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return [{
        "항목": "",
        "본문": "",
        "Risk": "",
        "위험요인": "",
        "개선사항": ""
    }]

# -----------------------------
# OpenAI 호출
# -----------------------------
def review_item(client, item_obj, prompt_text):
    item_text = item_obj["text"]
    page_num = item_obj["page"]

    for attempt in range(3):
        try:
            print(f"[OpenAI 요청] 페이지 {page_num}, 시도 {attempt+1}")
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": item_text}
                ],
                temperature=0.2,
                timeout=30,
                response_format={"type": "json_object"}
            )

            if not response or not response.choices:
                raise ValueError("응답이 비어있거나 choices가 없음")

            result_text = response.choices[0].message.content
            parsed = safe_json_loads(result_text)

            if not parsed or not isinstance(parsed, list):
                raise ValueError("JSON 파싱 실패 또는 결과 형식 오류")

            for entry in parsed:
                entry["Page"] = page_num

            print(f"[완료] 페이지 {page_num} 분석 성공")
            return parsed

        except Exception as e:
            print(f"❌ OpenAI 오류 (페이지 {page_num}, 시도 {attempt+1}): {e}")
            time.sleep(1.5)

    print(f"❌ 항목 처리 실패: {item_text[:30]}...")
    return []

# -----------------------------
# 병렬 처리
# -----------------------------
def process_items_parallel(client, items, prompt_text, file_path):
    results = []
    total = len(items)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_item = {
            executor.submit(review_item, client, item_obj, prompt_text): item_obj
            for item_obj in items
        }

        for i, future in enumerate(as_completed(future_to_item), start=1):
            try:
                item_result = future.result(timeout=60)
                if isinstance(item_result, list):
                    results.extend(item_result)
                else:
                    print(f"⚠️ 응답이 리스트가 아님: {item_result}")
            except Exception as e:
                print(f"❌ 항목 처리 실패: {e}")

            percent = int((i / total) * 100)
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (total - i)

            print(f"[진행률] {i} / {total} ({percent}%) | 예상 남은 시간: {int(remaining)}초")

    print("✅ 검토 완료")
    save_to_excel(results, file_path)

# -----------------------------
# 엑셀 저장
# -----------------------------
def save_to_excel(data, filename):
    if not data:
        print("저장할 데이터가 없습니다.")
        return

    def is_valid_row(row):
        if not isinstance(row, dict):
            return False
        본문 = row.get("본문", "")
        return isinstance(본문, str) and 본문.strip()

    clean_data = [row for row in data if is_valid_row(row)]
    print(f"필터링 후 유효한 항목 수: {len(clean_data)}")

    if not clean_data:
        print("유효한 데이터가 없습니다.")
        return

    os.makedirs("output", exist_ok=True)
    df = pd.DataFrame(clean_data, columns=["항목", "Page", "본문", "Risk", "위험요인", "개선사항"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join("output", f"review_{os.path.splitext(os.path.basename(filename))[0]}_{timestamp}.xlsx")
    df.to_excel(output_file, index=False, engine="openpyxl")
    print(f"📁 검토 결과 저장 완료: {output_file}")

# -----------------------------
# Main 실행
# -----------------------------
def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--file", help="검토할 문서 경로")
        args = parser.parse_args()

        file_path = args.file if args.file else select_contract_file()
        if not file_path:
            print("문서 선택 안됨. 종료합니다.")
            return

        items = read_pdf(file_path) if file_path.endswith(".pdf") else read_docx(file_path)
        if not items:
            print("문서에 분석할 항목이 없습니다.")
            return

        print(f"총 항목 수: {len(items)}")
        prompt_text = read_prompt()
        client = OpenAI(api_key=API_KEY)

        process_items_parallel(client, items, prompt_text, file_path)

    except Exception as e:
        print(f"실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()

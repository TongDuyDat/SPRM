import os
import json
import argparse
import time
from tqdm import tqdm

# Hỗ trợ cả hai thư viện phổ biến: google-generativeai và openai
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ==========================================
# 1. DANH SÁCH LỚP (CLASS NAMES) MẪU CHO 5 BỘ DỮ LIỆU
# ==========================================
DATASET_CLASSES = {
    "apy": [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor", "donkey", "monkey", "goat", "wolf", "jetski", "zebra", "centaur",
        "mug", "building", "bag", "carriage", "wheelchair"
    ],
    "awa1": [
        "antelope", "grizzly_bear", "killer_whale", "beaver", "dalmatian", "persian_cat", "horse", 
        "german_shepherd", "blue_whale", "siamese_cat", "skunk", "mole", "tiger", "hippopotamus", 
        "leopard", "spider_monkey", "humpback_whale", "elephant", "gorilla", "ox", "fox", "sheep", 
        "seal", "chimpanzee", "hamster", "squirrel", "rhinoceros", "rabbit", "bat", "giraffe", 
        "wolf", "chihuahua", "rat", "weasel", "otter", "buffalo", "zebra", "giant_panda", "deer", 
        "bobcat", "pig", "lion", "mouse", "polar_bear", "collie", "walrus", "raccoon", "cow", 
        "dolphin", "squirrel"
    ],
    "awa2": [
        "antelope", "grizzly_bear", "killer_whale", "beaver", "dalmatian", "persian_cat", "horse", 
        "german_shepherd", "blue_whale", "siamese_cat", "skunk", "mole", "tiger", "hippopotamus", 
        "leopard", "spider_monkey", "humpback_whale", "elephant", "gorilla", "ox", "fox", "sheep", 
        "seal", "chimpanzee", "hamster", "squirrel", "rhinoceros", "rabbit", "bat", "giraffe", 
        "wolf", "chihuahua", "rat", "weasel", "otter", "buffalo", "zebra", "giant_panda", "deer", 
        "bobcat", "pig", "lion", "mouse", "polar_bear", "collie", "walrus", "raccoon", "cow", 
        "dolphin", "squirrel"
    ],
    "cub": [
        # Ví dụ 20 lớp đại diện trong số 200 loài chim của CUB-200-2011.
        # Bạn có thể thay thế bằng đầy đủ danh sách 200 loài chim từ file classes.txt của CUB.
        "Black_footed_Albatross", "Laysan_Albatross", "Sooty_Albatross", "Groove_billed_Ani", 
        "Crested_Auklet", "Least_Auklet", "Parakeet_Auklet", "Rhinoceros_Auklet", "Brewer_Blackbird", 
        "Red_winged_Blackbird", "Rusty_Blackbird", "Yellow_headed_Blackbird", "Bobolink", "Indigo_Bunting", 
        "Lazuli_Bunting", "Painted_Bunting", "Cardinal", "Spotted_Towhee", "Olive_Warbler", "Yellow_Warbler"
    ],
    "sun": [
        # Ví dụ 20 lớp đại diện trong số 717 cảnh của SUN.
        # Bạn có thể thay thế bằng đầy đủ danh sách 717 cảnh từ file categories.txt của SUN.
        "abbey", "airport_terminal", "alley", "amphitheater", "amusement_park", "apartment_building", 
        "aqueduct", "arch", "art_gallery", "art_studio", "assembly_line", "attic", "auditorium", 
        "badlands", "ballroom", "bamboo_forest", "bank", "banquet_hall", "bar", "baseball_field"
    ]
}

# ==========================================
# 2. HÀM TẠO PROMPT CHO LLM
# ==========================================
def get_prompt(class_name, dataset_name):
    # Thay thế dấu gạch dưới thành dấu cách
    readable_name = class_name.replace("_", " ")
    
    prompt = f"""You are a helpful AI assistant specialized in semantic knowledge acquisition for zero-shot learning.
Your task is to generate highly comprehensive structural text descriptions for the class '{readable_name}' in the context of the '{dataset_name.upper()}' dataset. 

We want to achieve a very high density of descriptions (at least 120 sentences per class) similar to the original MS-COCO zero-shot learning post-LLM dataset.

We need exactly three types of descriptions formulated as simple sentences:
1. Hierarchical Relations (Generate about 60-80 combinations):
   - Subclasses/Variations (Hyponyms): Generate multiple sentences of the form: "a [subclass], which is a {readable_name}." (e.g., "a student, which is a person.")
   - Ancestors/Superclasses (Hypernyms): Generate multiple sentences of the form: "a {readable_name}, which is a [ancestor]." (e.g., "a person, which is a living organism.")
   - Multi-level phyla: Generate combinations of each subclass and ancestor of the form: "a [subclass], which is a {readable_name}, which is a [ancestor]." (e.g., "a student, which is a person, which is a living organism.")

2. Visual Attributes (Generate about 30-40 descriptions):
   - Description of physical parts, colors, shapes, textures, sizes, and typical properties: "a {readable_name}, which has [attribute description]." (e.g., "a person, which has hair: color (blonde, brown), style (curly).")

3. Co-occurrence Scenes (Generate about 15-20 descriptions):
   - Typical environmental surroundings, backgrounds, or situations where this class appears: "a {readable_name}, which appears together with the [environment/background description]." (e.g., "a person, which appears together with the Urban Settings")

### Instructions:
- Output ONLY a JSON array containing the candidate sentences as strings.
- Do NOT output any markdown tags (like ```json), no explanations, and no conversation. Only output raw JSON list.
- Generate at least 120 distinct, high-quality, and diverse candidate sentences covering all the categories above comprehensively.
- Ensure exact syntax patterns:
  - "a [X], which is a {readable_name}."
  - "a {readable_name}, which is a [Y]."
  - "a [X], which is a {readable_name}, which is a [Y]."
  - "a {readable_name}, which has [Z]."
  - "a {readable_name}, which appears together with the [W]."

Example output format:
[
  "a grizzly bear, which is a bear.",
  "a bear, which is a mammal.",
  "a grizzly bear, which is a bear, which is a mammal.",
  "a bear, which has thick fur: color (brown, black), which protects it from cold.",
  "a bear, which appears together with the Forest and River Settings"
]
"""
    return prompt

# ==========================================
# 3. KẾT NỐI API VÀ YÊU CẦU LLM
# ==========================================
def call_llm(prompt, provider, api_key, model_name=None):
    if provider == "gemini":
        if not GEMINI_AVAILABLE:
            raise ImportError("Vui lòng cài đặt google-generativeai: pip install google-generativeai")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name or "gemini-1.5-flash")
        response = model.generate_content(prompt)
        text = response.text.strip()
    elif provider == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError("Vui lòng cài đặt openai: pip install openai")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        text = response.choices[0].message.content.strip()
    else:
        raise ValueError("Provider không hợp lệ. Hãy chọn 'gemini' hoặc 'openai'.")
    
    # Làm sạch chuỗi kết quả nhận được (bỏ ```json ... ```)
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    
    try:
        sentences = json.loads(text)
        if isinstance(sentences, list):
            return sentences
    except Exception as e:
        print(f"Lỗi phân tích JSON từ phản hồi LLM: {e}")
        print(f"Nội dung phản hồi thô:\n{text}")
    
    return []

# ==========================================
# 4. CHƯƠNG TRÌNH CHÍNH (MAIN PROCESS)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Tự động sinh mô tả cấu trúc phân cấp và thuộc tính trực quan cho 5 bộ dữ liệu ZSL.")
    parser.add_argument("--dataset", type=str, required=True, choices=["awa1", "awa2", "cub", "sun", "apy"], help="Tên bộ dữ liệu cần tạo.")
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini", "openai"], help="API Provider sử dụng.")
    parser.add_argument("--api_key", type=str, required=True, help="API Key tương ứng.")
    parser.add_argument("--model", type=str, default=None, help="Tên model (Mặc định: 'gemini-1.5-flash' hoặc 'gpt-4o-mini').")
    parser.add_argument("--class_file", type=str, default=None, help="Đường dẫn đến file classes.txt chứa toàn bộ tên lớp nếu có (đọc từng dòng).")
    
    args = parser.parse_args()
    
    dataset_name = args.dataset.lower()
    
    # Xác định danh sách lớp
    if args.class_file and os.path.exists(args.class_file):
        print(f"Đang đọc danh sách lớp từ file: {args.class_file}")
        with open(args.class_file, "r") as f:
            classes = [line.strip().split()[-1] for line in f.readlines() if line.strip()]
    else:
        classes = DATASET_CLASSES[dataset_name]
        print(f"Sử dụng danh sách lớp mẫu của hệ thống cho {dataset_name} ({len(classes)} lớp).")
    
    output_dir = "redefine_classname"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}_post_llm.json")
    
    # Nạp dữ liệu cũ để tránh mất mát nếu tiến trình bị gián đoạn (Resume training)
    result_data = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                result_data = json.load(f)
            print(f"Tìm thấy tệp kết quả cũ. Đã tải {len(result_data)} lớp thành công.")
        except Exception:
            result_data = {}

    print(f"Bắt đầu sinh mô tả qua LLM ({args.provider}) cho bộ dữ liệu {dataset_name.upper()}...")
    
    for idx, class_name in enumerate(tqdm(classes)):
        str_idx = str(idx)
        
        # Nếu lớp này đã có dữ liệu rồi thì bỏ qua
        if str_idx in result_data and result_data[str_idx].get("candidate_sentences"):
            continue
            
        prompt = get_prompt(class_name, dataset_name)
        
        # Thử lại tối đa 3 lần nếu gặp lỗi API hoặc định dạng
        max_retries = 3
        sentences = []
        for attempt in range(max_retries):
            try:
                sentences = call_llm(prompt, args.provider, args.api_key, args.model)
                if sentences:
                    break
            except Exception as e:
                print(f"\n[Lớp {class_name}] Lỗi gọi API lần {attempt + 1}: {e}")
                time.sleep(2)
        
        if sentences:
            result_data[str_idx] = {
                "node_name": class_name,
                "candidate_sentences": sentences
            }
            # Lưu liên tục sau mỗi bước để phòng lỗi gián đoạn
            with open(output_file, "w") as f:
                json.dump(result_data, f, indent=4)
        else:
            print(f"\n[CẢNH BÁO] Không thể sinh mô tả cho lớp: {class_name}")
            
        time.sleep(1) # Tránh bị rate-limit
        
    print(f"\n Hoàn tất! Tệp mô tả đã được ghi vào: {output_file}")
    print(f"Tổng số lớp đã lưu: {len(result_data)}")

if __name__ == "__main__":
    main()

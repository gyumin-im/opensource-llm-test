import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. 모델 및 토크나이저 로드 (Qwen1.5 1.8B Chat)
model_id = "Qwen/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. 텍스트 생성 파이프라인 구축
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7
)

# 3. 대화 테스트 함수
def ask_llm(user_question):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Answer in Korean."},
        {"role": "user", "content": user_question}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, do_sample=True, top_k=50, top_p=0.95)
    response = outputs[0]["generated_text"][len(prompt):]
    print(f"👤 질문: {user_question}")
    print(f"🤖 LLM 답변: {response.strip()}\n")

# 4. 테스트 실행
ask_llm("안녕하세요! 오픈소스 LLM이 무엇인지 간단하게 설명해주세요.")
ask_llm("파이썬에서 회귀 분석을 할 때 사용하는 대표적인 라이브러리는 무엇인가요?")

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 환경 변수 로드
load_dotenv()

# 2. LLM 초기화

# OpenAI
# llm = ChatOpenAI(
#     model="gpt-4o-mini",  # 또는 gpt-3.5-turbo
#     temperature=0
# )

# Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # 또는 gemini-1.5-pro
    temperature=0
)

# 3. 프롬프트 1: 텍스트에서 사양 추출 (정보 추출)
prompt_extract = ChatPromptTemplate.from_template(
    "다음 텍스트에서 기술 사양을 추출하세요:\n\n{text_input}"
)

# 4. 프롬프트 2: JSON 변환
prompt_transform = ChatPromptTemplate.from_template(
    "다음 사양을 JSON 객체로 변환하세요.\n"
    "'CPU', '메모리', '저장소'를 키로 사용하세요.\n\n{specifications}"
)

# 5. 체인 구성 (LCEL)
extraction_chain = prompt_extract | llm | StrOutputParser()
# 전체 체인은 추출 체인의 출력을 변환 프롬프트의 'specifications' 변수에 전달
full_chain = (
    {"specifications": extraction_chain}
    | prompt_transform
    | llm
    | StrOutputParser()
)

# 6. 입력 텍스트
input_text = (
    "새로운 노트북 모델은 3.5GHz 옥타코어 프로세서, 16GB RAM, 1TB NVMe SSD를 탑재하고 있습니다."
)

# 7. 실행
final_result = full_chain.invoke({"text_input": input_text})

print("\n--- 최종 JSON 출력 ---")
print(final_result)

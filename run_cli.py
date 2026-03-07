from src.retriever import retrieve
from src.prompt import build_prompt, build_citation
from src.llm import call_llm
from src.tts import text_to_mp3

def main():
    while True:
        query = input("请输入你的问题（输入 q 退出）：").strip()
        if not query:
            continue
        if query.lower() == "q":
            break

        contexts = retrieve(query)
        if not contexts:
            print("\n未在知识库中检索到足够相关的内容，请换个问法或补充展品名称。")
            continue

        print("\n【检索到的知识】")
        for text, score in contexts:
            print(f"- {text} (score={score:.4f})")

        prompt = build_prompt(query, contexts)
        answer = call_llm(prompt)

        citation = build_citation(contexts)
        final_text = answer.strip() + (("\n\n" + citation) if citation else "")

        print("\n【生成的讲解】")
        print(final_text)

        mp3_path = text_to_mp3(final_text, filename="output.mp3")
        print("语音已生成：", mp3_path)

if __name__ == "__main__":
    main()

from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summerizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decision, extract_questions
from core.rag_engine import build_rag_chain, ask_question


load_dotenv()


def run_pipeline(source: str, language: str = "english") -> dict:
    print("Starting AI Videos Assistant")

    chunks = process_input(source=source)
    transcript = transcribe_all(chunks=chunks, language=language)
    print(f"Raw Trascription (first 300 character) {transcript[:300]}")

    title = generate_title(transcript=transcript)

    summary = summarize(transcript=transcript)

    action_item = extract_action_items(transcript=transcript)
    decision = extract_key_decision(transcript=transcript)
    questions = extract_questions(transcript=transcript)

    rag_chain = build_rag_chain(transcript=transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_item": action_item,
        "decision": decision,
        "questions": questions,
        "rag_chain": rag_chain
    }



if __name__ == "__main__":
    # CLI Entry Point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (enlihs/hinglish)").strip() or "english"
    result = run_pipeline(source=source, language=language)

    print("\n" + "=" * 60)
    print(f"Title: {result['title']}")
    print(f"Summary:\n {result['summary']}")
    print(f"Action Item:\n {result['action_item']}")
    print(f"Key Decision:\n {result['decision']}")
    print(f"Questions: {result['questions']}")


    print("\n Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result['rag_chain']
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye")
            break
        if not question:
            continue

        answer = ask_question(rag_chain, question)
        print(f"\n Assistant: {answer} \n")
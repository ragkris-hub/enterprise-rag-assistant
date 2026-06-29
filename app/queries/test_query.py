from app.services.rag_service import ask_question


if __name__ == "__main__":
    question = "What approval is required for production access?"

    response = ask_question(question)

    print("\n===== Question =====")
    print(response["question"])

    print("\n===== Answer =====")
    print(response["answer"])

    print("\n===== Citations =====")
    print(response["citations"])

    print("\n===== Retrieved Chunks =====")
    for index, chunk in enumerate(response["retrieved_chunks"], start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk["content"])
        print(chunk["metadata"])

# from app.services.rag_service import ask_question
# import pprint

# if __name__ == "__main__":
#     question = "What approval is required for production access?"

#     response = ask_question(question)

#     print("\n===== RAW RESPONSE =====")
#     pprint.pprint(response)
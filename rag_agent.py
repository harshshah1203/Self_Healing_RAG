import os
from typing import Literal, TypedDict

import chromadb
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

load_dotenv()

MAX_RETRIES = 2


class AgentState(TypedDict, total=False):
    question: str
    rewritten_question: str
    documents: list[str]
    answer: str
    grade: Literal["grounded", "not_grounded"]
    retry_count: int


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="knowledge_base")


def retrieve(state: AgentState) -> AgentState:
    """Search ChromaDB for documents relevant to the current question."""
    question = state.get("rewritten_question") or state["question"]

    print(f"\n[RETRIEVE] Searching for: {question}")

    try:
        results = collection.query(
            query_texts=[question],
            n_results=3,
        )
        documents = results["documents"][0] if results.get("documents") else []
        print(f"[RETRIEVE] Found {len(documents)} chunks.")
    except Exception as e:
        print(f"[RETRIEVE] ChromaDB query failed: {e}")
        documents = []

    return {**state, "documents": documents}


def generate(state: AgentState) -> AgentState:
    """Generate an answer using retrieved documents as context."""
    question = state.get("rewritten_question") or state["question"]
    documents = state.get("documents", [])

    print(f"\n[GENERATE] Generating answer for: {question}")

    if not documents:
        return {**state, "answer": "No relevant documents were found."}

    context = "\n\n".join(
        f"Document {i + 1}:\n{doc}" for i, doc in enumerate(documents)
    )

    system_prompt = (
        "You are a helpful assistant that answers questions based strictly on "
        "the provided documents. If the documents do not contain enough "
        "information to answer the question, say so honestly. Do not use any "
        "knowledge outside of the provided documents."
    )

    user_prompt = f"""Documents:
{context}

Question:
{question}

Answer the question based on the above documents."""

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        answer = response.content
        print(f"[GENERATE] Answer generated ({len(answer)} chars).")
    except Exception as e:
        print(f"[GENERATE] LLM call failed: {e}")
        answer = "An error occurred while generating the answer."

    return {**state, "answer": answer}


def grade_answer(state: AgentState) -> AgentState:
    """Grade whether the answer is grounded in the retrieved documents."""
    question = state.get("rewritten_question") or state["question"]
    documents = state.get("documents", [])
    answer = state.get("answer", "")

    print("\n[GRADE] Evaluating answer quality...")

    if not documents or not answer:
        print("[GRADE] not_grounded")
        return {**state, "grade": "not_grounded"}

    context = "\n\n".join(documents)
    grading_prompt = f"""You are a strict grader.

Decide whether the answer is fully grounded in the provided documents and answers the user's question.

Return exactly one word:
grounded
not_grounded

Documents:
{context}

Question:
{question}

Answer:
{answer}"""

    try:
        response = llm.invoke([HumanMessage(content=grading_prompt)])
        grade_text = response.content.strip().lower()
        grade: Literal["grounded", "not_grounded"] = (
            "grounded" if grade_text == "grounded" else "not_grounded"
        )
        print(f"[GRADE] {grade}")
    except Exception as e:
        print(f"[GRADE] LLM grading failed: {e}")
        grade = "not_grounded"

    return {**state, "grade": grade}


def rewrite_question(state: AgentState) -> AgentState:
    """Rewrite the user question to improve retrieval for another attempt."""
    retry_count = state.get("retry_count", 0) + 1
    original_question = state["question"]
    answer = state.get("answer", "")

    print(f"\n[REWRITE] Retrying with improved question ({retry_count}/{MAX_RETRIES})...")

    rewrite_prompt = f"""Rewrite this question for semantic document retrieval.

Keep the same intent, make it specific, and do not answer it.
Return only the rewritten question.

Original question:
{original_question}

Previous weak answer:
{answer}"""

    try:
        response = llm.invoke([HumanMessage(content=rewrite_prompt)])
        rewritten_question = response.content.strip()
    except Exception as e:
        print(f"[REWRITE] LLM rewrite failed: {e}")
        rewritten_question = original_question

    print(f"[REWRITE] New query: {rewritten_question}")

    return {
        **state,
        "rewritten_question": rewritten_question,
        "retry_count": retry_count,
    }


def decide_next_step(state: AgentState) -> str:
    """Route to finish or retry based on grading."""
    if state.get("grade") == "grounded":
        return "end"

    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "end"

    return "rewrite"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("grade_answer", grade_answer)
    graph.add_node("rewrite_question", rewrite_question)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grade_answer")
    graph.add_conditional_edges(
        "grade_answer",
        decide_next_step,
        {
            "rewrite": "rewrite_question",
            "end": END,
        },
    )
    graph.add_edge("rewrite_question", "retrieve")

    return graph.compile()


app = build_graph()


def ask(question: str) -> str:
    """Run the self-healing RAG agent and return the final answer."""
    final_state = app.invoke(
        {
            "question": question,
            "retry_count": 0,
        }
    )
    return final_state.get("answer", "")


if __name__ == "__main__":
    user_question = input("Ask a question: ").strip()
    if not user_question:
        print("Please ask a non-empty question.")
    else:
        print("\nFinal answer:")
        print(ask(user_question))

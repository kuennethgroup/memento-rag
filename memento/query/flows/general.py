"""General and chitchat messages (no question about the documents): the prompt for one direct answer, with the date,
the answering model and who built Memento, and the app's own links in the reply.
"""
import os
import re

from memento.query import generation, presentation, texts


# Router label "general": a message no document could answer (date/time, arithmetic with all numbers given, how to use
# the app, rewording pasted text), answered directly without retrieval or citations.
GENERAL_SYSTEM_PROMPT = (
    "You are Memento, an assistant for a shared library of research documents. The user's latest message is not "
    "a question about the documents in the library (a greeting, thanks, or a general question), so answer it "
    "directly and briefly from general knowledge, in the language of the message; to a greeting, reply in one or two "
    "sentences and say what you can do. Now it is {now} (server time). Do not invent content of documents and do "
    "not cite sources. "
    "If the message does ask about the library's documents after all, say that they can ask it as a question about "
    "the library.\n\nHow Memento is used (for questions about the app; don't add steps beyond these): ask questions "
    "about the library in Library Chat, in English or German; every answer cites its sources as [N], and clicking "
    "one opens the source page. The start screen has buttons for common tasks: Summarize a document, Limitations & "
    "future work, and Draft a proposal. When asked what you can do or how to use Memento, name the start-screen "
    "buttons and both ways to use your own documents. There are two ways to use your own documents, and when asked "
    "how to add, "
    "attach or upload a document, name both: "
    "[Library Upload](/?model=memento-ingestion-prod) adds files to the library, which is shared by everyone: pick "
    "it in the model menu, attach the files with + → Upload Files and send (PDF, Word, PowerPoint, Excel), then ask "
    "about them in Library Chat. [Document Chat](/?model=memento-document-chat) answers questions about a file of "
    "your own without adding it: pick it in the model menu and attach the file with + → Upload Files; the file stays "
    "private and is deleted with the chat. The \"Thorough check\" switch next to + makes Memento take time to think "
    "the question through and check each statement against the sources; answers take longer. Asked which model you "
    "are or which model answers: this answer is written by {model}; each answer "
    "names the model that wrote it underneath (\"Answered by …\"), the local model or a cloud model picked in the model "
    "menu. Asked who built you or who made Memento: Memento was built {built_at}as part of a Master's thesis. Give only "
    "these facts: asked for more about the project or the people behind it (names, the thesis, contact), say that you "
    "have no further details and that the administrator can help. Asked how Memento works: it finds the passages in "
    "the library's documents closest to the question, and the answer model writes the answer from them and cites "
    "them.")


def general_prompt(now: str, model: str) -> str:
    """GENERAL_SYSTEM_PROMPT for one answer; who built Memento comes from MEMENTO_BUILT_BY (config/.env), if set."""
    built = os.getenv("MEMENTO_BUILT_BY", "").strip()
    return GENERAL_SYSTEM_PROMPT.format(now=now, model=model, built_at=f"at {built} " if built else "")


# Document Chat's own (greetings go to the model too): what it does, and whether a file is attached
DOC_CHAT_GENERAL_SYSTEM_PROMPT = (
    "You are Document Chat, part of Memento. It answers questions about the files the user attaches in this chat and cites the "
    "pages it uses; the files are read only on the chair's server by its local model, are not added to the shared "
    "library and are deleted with the chat. The user's latest message is not a question about the files (a greeting, "
    "thanks, or a general question), so answer it directly and briefly from general knowledge, in the language of "
    "the message; to a greeting, reply in one or two sentences and say what you can do. Now it is {now} (server "
    "time). Do not invent content of files and do not cite sources. {files}")


async def reply(turn):
    """A greeting, thanks or a question not about the library: one answer-model call without thinking, no search, no
    citations; a general answer is marked as not from the library."""
    await turn.status("Answering directly...")
    from datetime import datetime
    now = datetime.now().astimezone().strftime("%A, %d %B %Y, %H:%M %Z")
    if turn.doc_chat is None:
        system = general_prompt(now, turn.variant.label)
    else:
        names = ", ".join(sorted(turn.doc_chat.names.values()))
        system = DOC_CHAT_GENERAL_SYSTEM_PROMPT.format(
            now=now, files=f"Files in this chat: {names}." if names else texts.DOC_CHAT_FILES_NONE)
    try:
        chunks, answered, note = await generation.open_answer(
            turn.valves, turn.variant, think=False, max_tokens=2048, budget=turn.budget,
            messages=[{"role": "system", "content": system}]
                     + turn.history + [{"role": "user", "content": turn.original_query}])
        text = await generation.drain_content(chunks, answered.is_cloud)
        text = presentation.link_app_names(re.sub(r"(?s)<think>.*?</think>", "", text).strip())
    except Exception as e:
        yield texts.user_error("answer", e)
        await turn.status("Done", done=True)
        turn.done = True
        return
    import logging
    logging.getLogger("memento.general").info("general answer, %d chars", len(text))
    marker = "" if turn.query_intent == "chitchat" else texts.GENERAL_NOTE if turn.doc_chat is None else texts.DOC_CHAT_GENERAL_NOTE
    yield note + (text or "I can't answer that one.") + marker + generation.answered_by(answered.label)
    await turn.status("Done", done=True)
    turn.done = True

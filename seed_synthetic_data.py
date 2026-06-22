"""Wipe MongoDB + Pinecone and reseed with synthetic data.

DESTRUCTIVE. Run with --confirm to actually wipe and reseed:

    KMP_DUPLICATE_LIB_OK=TRUE python seed_synthetic_data.py --confirm

Builds:
  - a synthetic mental-health Q&A corpus in PatientConvo + Pinecone (so the
    assistant's "similar cases" grounding still works),
  - ~N synthetic patients (profiles) in `patients`,
  - 2-4 past sessions per patient in `sessions` (matching the current schema),
  - one short archived conversation per session in PatientConvo.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import random
import uuid
from datetime import datetime, timedelta

from db import get_db
from model_cache import get_embedding_model, get_pinecone_index

random.seed(42)

CONDITIONS = {
    "anxiety": {
        "history": ["generalized anxiety disorder", "panic attacks", "chronic worry"],
        "goals": ["reduce daily anxiety", "manage panic episodes", "build coping skills"],
        "topics": ["anxiety", "stress", "sleep-improvement"],
    },
    "depression": {
        "history": ["major depressive disorder", "persistent low mood", "loss of interest"],
        "goals": ["rebuild motivation", "re-engage socially", "improve daily routine"],
        "topics": ["depression", "self-esteem", "grief-and-loss"],
    },
    "grief": {
        "history": ["recent bereavement", "complicated grief", "low mood after loss"],
        "goals": ["process the loss", "re-engage with daily life", "find support"],
        "topics": ["grief-and-loss", "depression", "relationships"],
    },
    "trauma": {
        "history": ["PTSD", "childhood trauma", "hypervigilance"],
        "goals": ["process past trauma safely", "reduce flashbacks", "feel safe"],
        "topics": ["trauma", "anxiety", "sleep-improvement"],
    },
    "stress": {
        "history": ["work-related stress", "burnout", "tension headaches"],
        "goals": ["set boundaries", "manage workload stress", "improve work-life balance"],
        "topics": ["stress", "workplace-relationships", "sleep-improvement"],
    },
    "relationships": {
        "history": ["relationship conflict", "communication difficulties"],
        "goals": ["improve communication", "rebuild trust", "set healthy boundaries"],
        "topics": ["relationships", "family-conflict", "self-esteem"],
    },
    "self-esteem": {
        "history": ["low self-esteem", "negative self-talk", "perfectionism"],
        "goals": ["build self-compassion", "challenge negative thoughts"],
        "topics": ["self-esteem", "anxiety", "depression"],
    },
    "anger": {
        "history": ["anger-management difficulties", "irritability"],
        "goals": ["manage anger triggers", "respond rather than react"],
        "topics": ["anger-management", "relationships", "stress"],
    },
    "sleep": {
        "history": ["chronic insomnia", "disrupted sleep"],
        "goals": ["improve sleep", "establish a bedtime routine"],
        "topics": ["sleep-improvement", "anxiety", "stress"],
    },
    "substance": {
        "history": ["alcohol use disorder", "substance misuse"],
        "goals": ["reduce substance use", "develop healthier coping"],
        "topics": ["substance-abuse", "depression", "anxiety"],
    },
}

# Building blocks for varied synthetic Q&A by topic.
_OPENERS = {
    "anxiety": ["I've been feeling anxious", "My anxiety has been overwhelming", "I keep worrying constantly", "I get panic attacks"],
    "depression": ["I feel empty and hopeless", "I've lost interest in everything", "I feel worthless most days", "I can't find motivation"],
    "grief-and-loss": ["I recently lost someone close", "I'm grieving and it hurts", "I can't stop thinking about my loss", "The grief feels unbearable"],
    "trauma": ["I keep reliving a traumatic event", "I have flashbacks", "I feel unsafe even when I'm fine", "My past trauma still haunts me"],
    "stress": ["Work stress is crushing me", "I'm completely burned out", "I feel overwhelmed by everything", "I can't switch off from work"],
    "relationships": ["My relationship is struggling", "We can't communicate", "I feel disconnected from my partner", "There's constant conflict at home"],
    "self-esteem": ["I feel like I'm not good enough", "My inner critic is relentless", "I doubt myself constantly", "I struggle with self-worth"],
    "anger-management": ["I get angry too easily", "I lose my temper often", "My irritability is hurting people", "I react before I think"],
    "sleep-improvement": ["I can't sleep at night", "My sleep is broken", "I lie awake for hours", "I wake up exhausted"],
    "substance-abuse": ["I'm drinking more than I should", "I rely on substances to cope", "I want to cut back but can't", "My substance use is escalating"],
    "family-conflict": ["My family relationships are tense", "We argue all the time at home", "I feel caught in family conflict"],
    "workplace-relationships": ["My workplace feels toxic", "I clash with my manager", "I dread going to work"],
}
_DETAILS = ["and it affects my sleep", "and I can't focus at work", "and it's straining my relationships",
            "especially in the mornings", "and I feel exhausted", "and it's getting worse",
            "and I don't know how to cope", "and I feel alone in it", "and it comes in waves"]
_ADVICE = {
    "default": [
        "Let's start by naming the feeling and noticing when it shows up. Tracking triggers in a journal often reveals patterns.",
        "Grounding techniques — slow breathing, the 5-4-3-2-1 senses exercise — can help in the moment.",
        "Be gentle with yourself; this is a common and treatable difficulty, and small consistent steps matter most.",
        "Consider building a simple daily routine and one supportive connection you can lean on.",
        "We can explore the thoughts underneath this feeling and gently test how accurate they are.",
        "Reaching out for support — a trusted person or a professional — is a strength, not a weakness.",
    ],
    "sleep-improvement": [
        "A consistent sleep schedule and a wind-down routine (no screens an hour before bed) often help.",
        "Try keeping the bedroom for sleep only, and get up if you can't sleep after 20 minutes.",
    ],
    "substance-abuse": [
        "Reducing use safely sometimes needs medical support; let's talk about what feels manageable.",
        "Identifying the need the substance is meeting helps us find healthier ways to meet it.",
    ],
    "trauma": [
        "Safety and pacing come first; we won't push faster than feels manageable.",
        "Grounding and a sense of present-moment safety are the foundation before processing the memory.",
    ],
}


def _wipe(db, index):
    print("Wiping MongoDB collections…")
    for coll in ("patients", "sessions", "PatientConvo"):
        res = db[coll].delete_many({})
        print(f"  cleared {coll}: {res.deleted_count}")
    print("Wiping Pinecone 'default' namespace…")
    try:
        index.delete(delete_all=True, namespace="default")
        print("  Pinecone default namespace cleared")
    except Exception as e:
        print("  Pinecone delete note:", repr(e))


def _build_corpus(db, index, embed, per_topic=14):
    print("Building synthetic Q&A corpus…")
    docs = []
    qid = 1
    topics = list(_OPENERS.keys())
    for topic in topics:
        seen = set()
        attempts = 0
        added = 0
        while added < per_topic and attempts < per_topic * 6:
            attempts += 1
            opener = random.choice(_OPENERS[topic])
            detail = random.choice(_DETAILS)
            question = f"{opener} {detail}."
            if question in seen:
                continue
            seen.add(question)
            answer_pool = _ADVICE.get(topic, []) + _ADVICE["default"]
            answer = " ".join(random.sample(answer_pool, k=min(2, len(answer_pool))))
            docs.append({
                "questionID": qid,
                "questionTitle": opener,
                "questionText": question,
                "answerText": answer,
                "topic": topic,
                "therapistInfo": "Synthetic Counselor, LCSW",
                "upvotes": random.randint(0, 40),
                "views": random.randint(20, 500),
            })
            qid += 1
            added += 1
    db["PatientConvo"].insert_many([dict(d) for d in docs])
    print(f"  inserted {len(docs)} corpus docs into PatientConvo")

    print("  embedding + upserting to Pinecone…")
    texts = [d["questionText"] for d in docs]
    vectors = embed.embed_documents(texts)
    batch = []
    for d, vec in zip(docs, vectors):
        batch.append({"id": f"q{d['questionID']}", "values": vec,
                      "metadata": {"questionID": d["questionID"], "topic": d["topic"],
                                   "questionTitle": d["questionTitle"]}})
    for i in range(0, len(batch), 100):
        index.upsert(vectors=batch[i:i + 100], namespace="default")
    print(f"  upserted {len(batch)} vectors to Pinecone")
    return len(docs)


def _build_patients(db, n_patients=50):
    print(f"Building {n_patients} synthetic patients + sessions…")
    cond_keys = list(CONDITIONS.keys())
    patients, sessions, conversations = [], [], []
    now = datetime.now()
    for i in range(1, n_patients + 1):
        pid = f"PT-{i:04d}"
        cond = CONDITIONS[random.choice(cond_keys)]
        mh = random.sample(cond["history"], k=min(2, len(cond["history"])))
        tg = random.sample(cond["goals"], k=min(2, len(cond["goals"])))
        patients.append({"patient_id": pid, "medical_history": mh, "therapy_goals": tg})

        n_sessions = random.randint(2, 4)
        for s in range(n_sessions):
            days_ago = (n_sessions - s) * random.randint(14, 40)
            created = now - timedelta(days=days_ago)
            sid = f"{pid}-S{s + 1}"
            topics = random.sample(cond["topics"], k=min(2, len(cond["topics"])))
            risk = ["suicide_risk"] if random.random() < 0.06 else []
            sentiment = round(random.uniform(-0.95, -0.2) + s * 0.2, 3)
            sessions.append({
                "session_id": sid, "patient_id": pid,
                "detected_topics": topics, "recommendations": [],
                "risk_flags": risk, "sentiment_score": sentiment,
                "doctor_notes": random.choice(["", "", "Patient engaged; reviewed coping plan.",
                                               "Discussed sleep hygiene and follow-up."]),
                "suggestions": [], "created_at": created,
            })
            conversations.append({
                "session_id": sid, "patient_id": pid,
                "messages": [
                    {"content": f"How have you been since our last session?", "is_user": False,
                     "speaker": "doctor", "timestamp": created, "metadata": {}},
                    {"content": f"{random.choice(_OPENERS[topics[0]])} this week.", "is_user": True,
                     "speaker": "patient", "timestamp": created, "metadata": {}},
                ],
                "created_at": created, "updated_at": created,
            })
    db["patients"].insert_many(patients)
    db["sessions"].insert_many(sessions)
    db["PatientConvo"].insert_many(conversations)
    print(f"  inserted {len(patients)} patients, {len(sessions)} sessions, {len(conversations)} conversations")
    return len(patients), len(sessions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="actually wipe + reseed")
    ap.add_argument("--patients", type=int, default=50)
    args = ap.parse_args()

    db = get_db()
    index = get_pinecone_index()

    if not args.confirm:
        print("DRY RUN — pass --confirm to wipe MongoDB + Pinecone and reseed.")
        print("Current counts:", {c: db[c].count_documents({}) for c in ("patients", "sessions", "PatientConvo")})
        return

    embed = get_embedding_model()
    _wipe(db, index)
    n_corpus = _build_corpus(db, index, embed)
    n_pat, n_sess = _build_patients(db, args.patients)

    print("\n=== DONE ===")
    print({c: db[c].count_documents({}) for c in ("patients", "sessions", "PatientConvo")})
    print(f"corpus docs: {n_corpus}, patients: {n_pat}, sessions: {n_sess}")


if __name__ == "__main__":
    main()

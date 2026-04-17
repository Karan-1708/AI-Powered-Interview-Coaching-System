# 🔭 Future Plans: What's Coming Next

The current system is already a capable interview coach — but we're not done. Below is the roadmap of features we're actively working towards. Each one is designed to push the system closer to the gold standard: an interview coach that's smarter, more private, and more realistic than anything else out there.

---

## 🏆 Hireability Readiness Score

*"How hire-ready are you, really?"*

After every interview, you currently get acoustic metrics and AI feedback — but no single number that captures the full picture. The **Hireability Readiness Score** changes that.

- At the end of each session, the system calculates a single percentage (e.g., *"82% hire-ready"*) that combines your acoustic performance (pacing, filler word count) with the AI's evaluation of how well your answers followed the STAR framework.
- The score is broken down into sub-scores so you know exactly where to improve — not just *that* you need to improve.
- Tracked across sessions so you can watch your number climb over time.

**Tech:** Weighted scoring formula over `AudioProcessor` acoustic output + structured LLM rubric evaluation via the `/generate-response` endpoint.

---

## 🎭 Multi-Agent Panel Interview

*"Face the whole room at once."*

Real panel interviews are brutal because every person in the room has a different agenda. The **Multi-Agent Panel Interview** mode brings that chaos to your practice sessions.

- Instead of one interviewer for the whole session, the system rotates between distinct AI personas turn-by-turn — a Friendly HR rep, a Strict Tech Lead, a Stress Interviewer, and more.
- Each persona picks up the conversation where the last one left off, forcing you to constantly shift your communication style mid-session.
- The closest simulation of a real multi-interviewer panel that doesn't require booking a conference room.

**Tech:** Session-level persona rotation state in `interview.py`, powered by the existing `Personas.get_interviewer_by_type()` routing — no new personas required, just a new orchestration mode.

---

## 🎵 Advanced Audio Sentiment & Emotion Analysis

*"It's not just what you say — it's how you say it."*

Right now the system hears your words. Soon, it'll hear your nerves.

- Using paralinguistic analysis on the raw audio file, the system will detect your emotional state — confidence, anxiety, hesitation — purely from vocal tone, pitch variance, and pause patterns.
- This operates entirely independently of the transcript, meaning it catches what the words alone can't show.
- Reported alongside your existing acoustic metrics in the Performance Review.

**Tech:** `librosa` MFCC and pitch-shift feature extraction added to the `AudioProcessor` pipeline, fed into a lightweight classifier to output a confidence/anxiety index.

---

## 🔒 Automated PII Redaction — *"Privacy Shield"*

*"Your personal details stay personal."*

Even with local-first storage, transcripts can contain sensitive information you didn't intend to keep — phone numbers, home addresses, identification numbers. The **Privacy Shield** handles this automatically.

- A Named Entity Recognition (NER) pass scans every transcript immediately after transcription and redacts Personally Identifiable Information before anything is saved or sent to the LLM.
- Zero configuration required — it runs silently in the background on every session.
- Designed to meet enterprise-grade data compliance standards, making the system suitable for institutional and organizational deployment.

**Tech:** `spaCy` NER pipeline injected as a post-processing step inside `AudioProcessor.process_interview()`, before transcript data leaves the audio engine.

---

## 🔍 Real-Time Company & Industry Research — *RAG Mode*

*"Ask the questions only that company would ask."*

Generic interview questions only get you so far. The most memorable candidates walk in knowing the company's recent news, values, and direction. **RAG Mode** gives you that edge automatically.

- Paste a company URL into the setup wizard, and the system scrapes recent articles, press releases, and core values pages in real time.
- That intelligence is fed directly into the question generation pipeline, so your interview includes timely, company-specific questions you couldn't have prepared for from a generic list.
- Pairs seamlessly with the existing Resume and Job Description context — the AI now knows your background *and* their company.

**Tech:** Retrieval-Augmented Generation pipeline — web scraper → chunked vector store → retrieval-augmented context injected into the `/generate-questions` endpoint.

---

## 💡 A Note on the Roadmap

These aren't vague ideas on a whiteboard — they're the natural next layer of a system we've already built carefully from the ground up. Each feature extends what's already there rather than rebuilding it.

If you're a developer and one of these features excites you, the codebase is structured to make it straightforward to plug in. The architecture is open, the components are modular, and the coffee is (presumably) still hot.

More to come. Stay tuned.

---

*Built by the Data Drifters.*
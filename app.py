"""
ParentPal - a 2 a.m. helpdesk for parents (CrewAI + Streamlit)
===============================================================

Three agents run SEQUENTIALLY for every parent question:

    1. Assistant              -> answers from its own knowledge base: WHO (www.who.int) pages
    2. Web Search Assistant   -> searches the web (official sources first) and answers
    3. Entry Agent            -> saves question + both answers to baby_journal.txt and
                                 returns both answers to the UI

Extras: emergency red-flag check (no AI involved), child-age aware answers, a
"Prepare for my doctor visit" sheet built from the journal, and token-saving design.

Run:
    export OPENAI_API_KEY="..."      # macOS / Linux
    export SERPER_API_KEY="..."
    streamlit run app.py

API keys are ONLY read from environment variables (or a git-ignored .env file).
ParentPal gives general information, not medical advice.
"""

from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import streamlit as st
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_TITLE = "ParentPal"
LOG_FILE = Path(os.getenv("PARENTPAL_JOURNAL", "baby_journal.txt"))
ENTRY_SEPARATOR = "=" * 80
MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"]
CHILD_AGES = ["Newborn (0-3 months)", "4-6 months", "7-11 months", "1-2 years", "3-4 years",
              "5+ years", "Pregnancy / expecting"]
SAMPLE_QUERIES = [
    "When should I start solid foods for my baby?",
    "How much sleep does my 8-month-old need?",
    "My toddler has diarrhoea. What should I do?",
    "Is it safe to give several vaccines in one visit?",
]
AGENTS_UI = [
    ("🧠", "Assistant", "Answers from WHO (who.int) knowledge"),
    ("🌐", "Web Search Assistant", "Checks trusted sources on the web"),
    ("📔", "Entry Agent", "Saves to your baby journal"),
]

# ---------------------------------------------------------------------------
# Emergency red flags - checked in plain Python BEFORE any AI runs (0 tokens, instant)
# ---------------------------------------------------------------------------
RED_FLAGS = [
    r"not breathing", r"can'?t breathe", r"stopped breathing", r"struggling to breathe",
    r"blue (lips|face|skin)", r"turning blue", r"seizure", r"convuls",
    r"unconscious", r"unresponsive", r"won'?t wake", r"not waking", r"\blimp\b", r"\bfloppy\b",
    r"swallowed (a )?(battery|magnet|poison|medicine|pills?|bleach|chemical|kerosene)", r"\bpoisoned\b",
    r"choking", r"severe bleeding", r"head injury", r"fell (from|off) .*(bed|stairs|height)",
    # Fever in a baby under 3 months always needs a doctor promptly.
    r"(fever|temperature).*(newborn|\b\d{1,2} ?(day|week)s?\b|\b[0-2] ?months?\b)",
    r"(newborn|\b\d{1,2} ?(day|week)s?\b|\b[0-2] ?months?\b).*(fever|temperature)",
    r"stiff neck", r"rash .*(doesn'?t|does not) fade", r"no wet nappies|no wet diapers",
]
EMERGENCY_NUMBERS = {
    "India": "112 (emergency) or 108 (ambulance)",
    "United States": "911",
    "United Kingdom": "999 (or 111 for urgent advice)",
    "Other": "your local emergency number",
}


def red_flag(text: str) -> str | None:
    t = text.lower()
    for pattern in RED_FLAGS:
        if re.search(pattern, t):
            return pattern
    return None


# ---------------------------------------------------------------------------
# Assistant knowledge base: WHO (www.who.int)
# ---------------------------------------------------------------------------
# who.int (the homepage) is fetched live for the latest WHO news. The parent-focused WHO
# pages below are the Assistant's core knowledge. A condensed snapshot of each page
# (taken 2026-10-10) is bundled so the app works even when a live fetch fails.
KB_SNAPSHOT_DATE = "2026-10-10"
KB_SOURCES = [
    {
        "title": "WHO homepage - latest news",
        "url": "https://www.who.int/",
        "live_first": True,  # changes daily, so always try live
        "snapshot": """## Latest WHO news (snapshot)
- 9 Oct 2026: WHO urges a shift from institutional to community-based mental health care.
- 8 Oct 2026: New global roadmap to tackle hypertension in pregnancy, a leading cause of maternal and newborn deaths.
- 7 Oct 2026: WHO issues its first global guidelines on child and adolescent obesity.
- 30 Sep 2026: New guidance to prevent heat-related illness at mass gatherings.
- 10 Oct is World Mental Health Day. 18-24 Oct is International Lead Poisoning Prevention Week.
""",
    },
    {
        "title": "Infant and young child feeding (WHO fact sheet)",
        "url": "https://www.who.int/news-room/fact-sheets/detail/infant-and-young-child-feeding",
        "snapshot": """## Breastfeeding recommendations
- WHO and UNICEF recommend starting breastfeeding within 1 hour of birth.
- Exclusive breastfeeding (breast milk only, not even water unless medically necessary) for the first 6 months.
- Then continue breastfeeding up to 2 years of age or beyond, alongside safe, nutritious solid foods.
- Breastfeed responsively: as often as the child wants, day and night. Skin-to-skin contact right after birth and rooming-in help.
- Breastfeeding protects against diarrhoea and pneumonia; breastfed children are less likely to become overweight. It also lowers the mother's risk of breast and ovarian cancer.
- Breast milk gives half or more of a child's energy needs at 6-12 months and about one third at 12-24 months, and is important during illness.
- Mothers living with HIV on lifelong ART can breastfeed with very little risk; they should breastfeed for at least 12 months.

## Starting solid foods (complementary feeding)
- Around 6 months, a baby needs more energy and nutrients than breast milk alone gives. Starting solids too early or too late can slow growth.
- Start at 6 months with small amounts and increase gradually; slowly increase thickness and variety.
- Meals per day: 2-3 meals at 6-8 months; 3-4 meals at 9-23 months, plus 1-2 snacks if needed.
- Feed responsively: notice and respond to signs of hunger and fullness.
- Practise good hygiene and safe food handling.
- During illness, give more fluids (including more breastfeeds) and soft, favourite foods.
""",
    },
    {
        "title": "Vaccines and immunization: What is vaccination? (WHO Q&A)",
        "url": "https://www.who.int/news-room/questions-and-answers/item/vaccines-and-immunization-what-is-vaccination",
        "snapshot": """## How vaccines work and why they matter
- Vaccines train the immune system to make antibodies using killed or weakened germs (or parts of them), so they do not cause the disease.
- Most are injections; some are given by mouth or nasal spray.
- Childhood vaccines save over 4 million lives every year (WHO estimate). Vaccinating also protects babies too young to be vaccinated.

## Vaccination timing and schedule
- Keep the child's vaccination card; it shows which vaccines were given and when the next doses are due. Keep vaccines up to date.
- Delaying vaccination leaves the child at risk. If doses were missed, talk to a health worker about catching up.
- The schedule differs by country - ask your health worker for your national schedule.

## Several vaccines in one visit and combination vaccines
- It is safe to give several different vaccines in one visit; combination vaccines (e.g. DTP) are as safe and effective as separate ones.
- Multiple vaccines do not overwhelm the immune system - the antigens are a tiny fraction of what the body meets every day.

## Who should not get certain vaccines, or should wait
- People with illnesses or treatments that weaken immunity, very rare severe allergies to an ingredient, or severe illness with high fever on the day. Ask your health worker.
- Vaccine ingredients: antigen, adjuvants (e.g. aluminium salts), preservatives (e.g. thiomersal) and stabilisers (e.g. sugar, gelatin) - all tested and monitored for safety.
- For reliable vaccine information online, use sites that are members of the WHO-reviewed Vaccine Safety Net.
""",
    },
    {
        "title": "Physical activity, screen time and sleep for children under 5 (WHO guidelines)",
        "url": "https://www.who.int/news/item/24-04-2019-to-grow-up-healthy-children-need-to-sit-less-and-play-more",
        "snapshot": """## Infants under 1 year: activity, screen time and sleep
- Active several times a day, especially floor-based play. Not yet mobile: at least 30 minutes of tummy time spread through the day while awake.
- Not restrained for more than 1 hour at a time (pram, high chair, carrier). Screen time is not recommended. Read and tell stories together.
- Sleep: 14-17 hours (0-3 months) or 12-16 hours (4-11 months) of good-quality sleep, including naps.

## Children 1-2 years: activity, screen time and sleep
- At least 180 minutes of varied physical activity spread through the day; more is better.
- Not restrained for more than 1 hour at a time. No screen time for 1-year-olds; for 2-year-olds no more than 1 hour - less is better.
- Sleep: 11-14 hours of good-quality sleep including naps, with regular sleep and wake-up times.

## Children 3-4 years: activity, screen time and sleep
- At least 180 minutes of activity, including at least 60 minutes of moderate-to-vigorous activity; more is better.
- Screen time no more than 1 hour a day - less is better.
- Sleep: 10-13 hours of good-quality sleep, which may include a nap, with regular sleep and wake-up times.
""",
    },
    {
        "title": "Diarrhoeal disease (WHO fact sheet)",
        "url": "https://www.who.int/news-room/fact-sheets/detail/diarrhoeal-disease",
        "snapshot": """## What counts as diarrhoea in children
- 3 or more loose or liquid stools a day (or more often than normal for the child).
- Loose, pasty stools in breastfed babies are NOT diarrhoea.
- Types: acute watery (hours to days), acute bloody (dysentery), persistent (14 days or longer).

## Dehydration warning signs
- Severe dehydration (2 or more): very sleepy or unconscious, sunken eyes, unable to drink or drinks poorly, skin pinch goes back very slowly (2 seconds or more).
- Some dehydration (2 or more): restless or irritable, sunken eyes, drinks eagerly / very thirsty.

## Treating diarrhoea at home and when to see a doctor
- Give oral rehydration solution (ORS: clean water, salt and sugar) to replace lost fluids.
- Zinc supplements for 10-14 days shorten diarrhoea and reduce stool volume (as advised by a health worker).
- Keep feeding nutrient-rich food, and keep breastfeeding during the illness.
- See a health professional for blood in the stool, diarrhoea lasting 14 days or more, or any signs of dehydration. Severe dehydration may need IV fluids.

## Preventing diarrhoea
- Safe drinking water, good sanitation, handwashing with soap, good food hygiene, exclusive breastfeeding for 6 months, and rotavirus vaccination.
""",
    },
]
KB_MAX_CHARS = 3000  # per question - only the relevant sections are sent


def _html_to_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "svg", "form"]):
            tag.decompose()
        root = soup.find("main") or soup.find(id="content") or soup.find("article") or soup.body or soup
        text = root.get_text("\n")
    except ImportError:
        text = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html)
        text = re.sub(r"<[^>]+>", "\n", text)
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if len(ln) > 2)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_page(url: str) -> tuple[str, str]:
    """Return (text, error). Cached for 6 hours."""
    try:
        import requests

        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 ParentPal/1.0"})
        resp.raise_for_status()
        text = _html_to_text(resp.text)
        return (text, "") if len(text) > 400 else ("", "page had no readable text")
    except Exception as exc:
        return "", str(exc)[:140]


def load_knowledge(sources: list[dict], live_all: bool) -> list[dict]:
    """Resolve every source to text: live page when wanted and available, else snapshot."""
    out = []
    for src in sources:
        text, mode, err = src.get("snapshot", ""), "snapshot", ""
        if src.get("live_first") or live_all or not text:
            live, err = fetch_page(src["url"])
            if live:
                text, mode = live[:6000], "live"
        out.append({**src, "text": text, "mode": mode if text else "unavailable", "error": err})
    return out


def build_chunks(kb: list[dict]) -> list[str]:
    """Split every source into small titled chunks for retrieval."""
    chunks = []
    for src in kb:
        if not src["text"]:
            continue
        head = f"[{src['title']}]({src['url']})"
        if src["mode"] == "snapshot":
            for sec in re.split(r"\n(?=## )", src["text"].strip()):
                chunks.append(f"{sec.strip()}\nSource: {head}")
        else:  # live text has no markdown headings -> ~1000 char pieces
            buf, part = "", 1
            for line in src["text"].splitlines():
                buf += line + "\n"
                if len(buf) > 1000:
                    chunks.append(f"## {src['title']} (live, part {part})\n{buf}Source: {head}")
                    buf, part = "", part + 1
            if buf.strip():
                chunks.append(f"## {src['title']} (live, part {part})\n{buf}Source: {head}")
    return chunks


_STOP = set(
    "a an the and or of to in on for with is are be it this that you your can if by as at "
    "from will not do does how what i my we our me should when much many baby babies child "
    "children kid kids old".split()
)


def _terms(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOP}


def retrieve(chunks: list[str], query: str, top_k: int = 3) -> str:
    """Tiny keyword retriever: the top_k chunks that best match the question."""
    q = _terms(query)

    def score(chunk: str) -> int:  # heading matches count triple
        head, _, body = chunk.partition("\n")
        return 3 * len(q & _terms(head)) + len(q & _terms(body))

    best = [c for c in sorted(chunks, key=score, reverse=True)[:top_k] if score(c) >= 2]
    return "\n\n".join(best)[:KB_MAX_CHARS]


# ---------------------------------------------------------------------------
# Run state + the Entry Agent's tool
# ---------------------------------------------------------------------------
_file_lock = threading.Lock()
_save_receipts: list[str] = []
_run_meta: dict = {}
_answers: dict[str, str] = {}  # filled as tasks finish -> saved by reference (no re-typing)


def write_entry(question: str, a1: str, a2: str, note: str = "") -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    block = (
        f"{ENTRY_SEPARATOR}\n"
        f"Timestamp : {stamp}\n"
        f"Child age : {_run_meta.get('age', '-')}\n"
        f"Model     : {_run_meta.get('model', '-')}{note}\n"
        f"{'-' * 80}\n"
        f"QUERY:\n{question.strip()}\n\n"
        f"ANSWER 1 - Assistant (WHO knowledge):\n{a1.strip()}\n\n"
        f"ANSWER 2 - Web Search Assistant (live web):\n{a2.strip()}\n"
    )
    with _file_lock:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(block + "\n")
    return str(LOG_FILE.resolve())


@tool("Save Journal Entry")
def save_journal_entry(question: str) -> str:
    """Save the parent's question with answer 1 (Assistant) and answer 2 (Web Search
    Assistant) into the baby journal .txt file. Only pass the question: both answers are
    attached automatically. Returns the saved file path."""
    path = write_entry(_run_meta.get("query", question), _answers.get("assistant", ""), _answers.get("web", ""))
    _save_receipts.append(path)
    return f"Saved question + answer 1 + answer 2 to {path}"


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def serper_search(q: str, n: int) -> str:
    """Compact Serper results (title, snippet, link). Cached for 6 hours."""
    import requests

    resp = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": os.getenv("SERPER_API_KEY", ""), "Content-Type": "application/json"},
        json={"q": q, "num": n},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    lines = []
    for i, r in enumerate(data.get("organic", [])[:n], 1):
        lines.append(f"{i}. {r.get('title', '')}\n   {r.get('snippet', '')[:280]}\n   {r.get('link', '')}")
    return "\n".join(lines) or "No results found."


@tool("Web Search")
def web_search(search_query: str) -> str:
    """Search the web (Google) and return the top results as short title / snippet / link
    entries. Use one focused query; adding a trusted site name (WHO, UNICEF, NHS, IAP,
    AAP HealthyChildren) gives better results."""
    try:
        return serper_search(search_query.strip(), int(_run_meta.get("n_results", 3)))
    except Exception as exc:
        return f"Search failed: {exc}"


# ---------------------------------------------------------------------------
# The crew
# ---------------------------------------------------------------------------
def build_crew(model: str, temperature: float, task_cb, step_cb) -> Crew:
    llm = LLM(model=model, temperature=temperature, max_tokens=450)
    llm_entry = LLM(model=model, temperature=0, max_tokens=120)

    assistant = Agent(
        role="Assistant",
        goal="Answer parents warmly and accurately from your WHO knowledge base.",
        backstory=(
            "A kind, experienced child-health educator who knows the World Health Organization's "
            "guidance on feeding, vaccines, sleep, activity and common illnesses. You never "
            "diagnose and never invent facts that are not in your knowledge base."
        ),
        llm=llm, allow_delegation=False, max_iter=2, verbose=False,
    )
    web_assistant = Agent(
        role="Web Search Assistant",
        goal="Answer parents from fresh, trustworthy web sources, citing them.",
        backstory=(
            "A careful researcher for parents. You prefer official health sources (WHO, UNICEF, "
            "national health services, paediatric associations) over blogs and forums."
        ),
        tools=[web_search], llm=llm, allow_delegation=False, max_iter=3, verbose=False,
    )
    entry_agent = Agent(
        role="Entry Agent",
        goal="Record every question in the baby journal and hand both answers back to the parent.",
        backstory="The family's record keeper. You save each question and confirm where it was saved.",
        tools=[save_journal_entry], llm=llm_entry, allow_delegation=False, max_iter=2, verbose=False,
    )

    answer_task = Task(
        description=(
            "Parent's question: '{query}'\nChild's age: {age}. Country: {country}.\n\n"
            "YOUR KNOWLEDGE BASE (World Health Organization, who.int):\n"
            "<<<KNOWLEDGE\n{kb}\nKNOWLEDGE>>>\n\n"
            "Answer ONLY from the knowledge base, adapted to the child's age, in warm, simple "
            "language. If it is not covered, start with 'This isn't covered in my WHO sources' and "
            "suggest asking a paediatrician. Never diagnose. Max 150 words, markdown, short "
            "steps where useful. If the question is about symptoms, add a 'See a doctor if:' line. "
            "End with 'Reference: <WHO page title>'."
        ),
        expected_output="A warm, concise markdown answer ending with a 'Reference:' line.",
        agent=assistant,
    )
    web_task = Task(
        description=(
            "Parent's question: '{query}'\nChild's age: {age}. Country: {country}.\n"
            "Run ONE web search (prefer official sources such as WHO, UNICEF, the national health "
            "service or paediatric association for {country}), then answer ONLY from the results "
            "(ignore any earlier answer you may have been shown). Warm, simple language, never "
            "diagnose. Max 150 words, markdown. End with 'Sources:' and the URLs you used."
        ),
        expected_output="A concise markdown answer ending with a 'Sources:' list of URLs.",
        agent=web_assistant,
        context=[],  # independent answer (no copy of answer 1)
    )
    entry_task = Task(
        description=(
            "Parent's question: '{query}'. Answer 1 (Assistant) and answer 2 (Web Search "
            "Assistant) are in your context. Call 'Save Journal Entry' ONCE with the question - "
            "both answers are attached automatically, so do not copy them. Then reply with one "
            "line: 'Saved to <path>'."
        ),
        expected_output="One line: 'Saved to <file path>'.",
        agent=entry_agent,
        context=[answer_task, web_task],  # both earlier answers flow into this task
    )

    def capture(output):
        key = ["assistant", "web", "entry"][min(len(_answers), 2)]
        _answers[key] = output.raw
        if task_cb:
            task_cb(output)

    return Crew(
        agents=[assistant, web_assistant, entry_agent],
        tasks=[answer_task, web_task, entry_task],
        process=Process.sequential,
        task_callback=capture,
        step_callback=step_cb,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Journal helpers
# ---------------------------------------------------------------------------
def read_journal() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    entries = []
    for chunk in LOG_FILE.read_text(encoding="utf-8").split(ENTRY_SEPARATOR):
        m = re.search(
            r"Timestamp : (.*?)\n(?:Child age : (.*?)\n)?.*?QUERY:\n(.*?)\n\nANSWER 1.*?:\n(.*?)\n\nANSWER 2.*?:\n(.*)",
            chunk.strip(), re.S,
        )
        if m:
            entries.append({"time": m[1].strip(), "age": (m[2] or "-").strip(), "query": m[3].strip(),
                            "a1": m[4].strip(), "a2": m[5].strip()})
    return entries[::-1]


def first_sentence(text: str, limit: int = 160) -> str:
    plain = re.sub(r"[#*_>`\[\]]", "", text).strip()
    plain = re.sub(r"\(https?://[^)]+\)", "", plain)
    sentence = re.split(r"(?<=[.!?])\s|\n", plain, maxsplit=1)[0]
    return sentence[:limit] + ("…" if len(sentence) > limit else "")


def visit_sheet(entries: list[dict], child_name: str) -> str:
    """Doctor-visit question list built from the journal - pure Python, 0 tokens."""
    who = child_name.strip() or "my child"
    lines = [f"Questions for the doctor about {who}", f"Prepared: {datetime.now():%d %b %Y}", "=" * 50, ""]
    for i, e in enumerate(reversed(entries), 1):
        lines += [f"{i}. [{e['time'][:10]} · {e['age']}] {e['query']}",
                  f"   What I read (WHO): {first_sentence(e['a1'])}",
                  "   Doctor's answer: ______________________________________", ""]
    lines += ["Note: ParentPal gives general information, not medical advice."]
    return "\n".join(lines)


_ASTOP = _STOP | set("all any but has have may more most only than then them they very".split())


def agreement_score(a: str, b: str) -> int:
    """Rough % of shared key terms between the two answers (overlap vs the shorter answer).
    Source lists, references and URLs are ignored so they don't count as disagreement."""
    def words(s: str) -> set[str]:
        s = re.sub(r"https?://\S+", " ", s)
        s = re.split(r"\n\s*\**\s*(sources|reference)\s*:?", s, flags=re.I)[0]
        return {w[:5] for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _ASTOP and len(w) > 2}
    wa, wb = words(a), words(b)
    return round(100 * len(wa & wb) / min(len(wa), len(wb))) if wa and wb else 0


def extract_sources(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)\]>\"']+", text)
    return list(dict.fromkeys(u.rstrip(".,;") for u in urls))[:6]


def domain(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", url).split("/")[0]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
CSS = """
<style>
.hero {padding:1.3rem 1.6rem;border-radius:20px;margin-bottom:1rem;color:#fff;
       background:linear-gradient(120deg,#f472b6 0%,#a78bfa 55%,#60a5fa 100%);}
.hero h1 {margin:0;font-size:2rem;color:#fff;} .hero p {margin:.3rem 0 0;opacity:.95;}
.step {border-radius:14px;padding:.8rem 1rem;border:1px solid rgba(128,128,128,.25);
       background:rgba(128,128,128,.06);}
.step.running {border-color:#f59e0b;background:rgba(245,158,11,.10);animation:pulse 1.4s infinite;}
.step.done {border-color:#10b981;background:rgba(16,185,129,.10);}
.step.error {border-color:#ef4444;background:rgba(239,68,68,.10);}
.step .t {font-weight:700;} .step .s {font-size:.82rem;opacity:.75;}
.badge {display:inline-block;padding:.1rem .55rem;border-radius:999px;font-size:.75rem;
        font-weight:600;margin-top:.35rem;background:rgba(128,128,128,.18);}
.src a {display:inline-block;margin:.2rem .3rem .2rem 0;padding:.15rem .6rem;border-radius:999px;
        font-size:.78rem;text-decoration:none;border:1px solid rgba(167,139,250,.6);}
.sos {border-radius:16px;padding:1.2rem 1.4rem;background:rgba(239,68,68,.12);
      border:2px solid #ef4444;margin:.6rem 0;}
.sos h3 {margin:0 0 .4rem;color:#ef4444;}
@keyframes pulse {0%{opacity:1}50%{opacity:.65}100%{opacity:1}}
</style>
"""


def render_pipeline(placeholder, states, timings) -> None:
    labels = {"waiting": "⏸ Waiting", "running": "⏳ Working…", "done": "✅ Done", "error": "❌ Error"}
    html = "<div style='display:flex;gap:.8rem;flex-wrap:wrap'>"
    for i, (icon, name, desc) in enumerate(AGENTS_UI):
        html += (f"<div class='step {states[i]}' style='flex:1;min-width:190px'>"
                 f"<div class='t'>{icon} {i + 1}. {name}</div><div class='s'>{desc}</div>"
                 f"<span class='badge'>{labels[states[i]]} {timings[i]}</span></div>")
    placeholder.markdown(html + "</div>", unsafe_allow_html=True)


def key_status(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


def run_crew(query, model, temperature, n_results, age, country, kb_text) -> dict:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

    ctx = get_script_run_ctx()
    states, timings = ["running", "waiting", "waiting"], ["", "", ""]
    progress = {"i": 0, "t0": time.time(), "tick": time.time()}
    feed: list[str] = []
    pipe_ph = st.empty()
    status = st.status("👶 The crew is looking into your question…", expanded=True)
    feed_ph = status.empty()
    render_pipeline(pipe_ph, states, timings)

    def _ui(fn):
        try:
            add_script_run_ctx(threading.current_thread(), ctx)
            fn()
        except Exception:
            pass

    def log(msg):
        feed.append(msg)
        _ui(lambda: feed_ph.markdown("\n\n".join(feed[-8:])))

    def task_cb(_output):
        i = progress["i"]
        now = time.time()
        timings[i], states[i], progress["tick"] = f"· {now - progress['tick']:.1f}s", "done", now
        log(f"✅ **{AGENTS_UI[i][1]}** finished")
        if i + 1 < 3:
            states[i + 1] = "running"
            log(f"▶️ **{AGENTS_UI[i + 1][1]}** started")
        progress["i"] = i + 1
        _ui(lambda: render_pipeline(pipe_ph, states, timings))

    def step_cb(step):
        if getattr(step, "tool", None):
            log(f"🔧 `{step.tool}` → {str(getattr(step, 'tool_input', ''))[:90]}")

    _save_receipts.clear()
    _answers.clear()
    _run_meta.update(model=model, query=query, age=age, n_results=n_results)
    log(f"▶️ **{AGENTS_UI[0][1]}** started")

    crew = build_crew(model, temperature, task_cb, step_cb)
    try:
        result = crew.kickoff(inputs={
            "query": query, "age": age, "country": country,
            "kb": kb_text or "(No matching WHO section for this question.)",
        })
    except Exception:
        states[min(progress["i"], 2)] = "error"
        render_pipeline(pipe_ph, states, timings)
        status.update(label="Something went wrong", state="error")
        raise

    a1, a2 = result.tasks_output[0].raw, result.tasks_output[1].raw
    if _save_receipts:
        saved_to = _save_receipts[-1]
    else:
        saved_to = write_entry(query, a1, a2, note="  (safety-net save)")
        log("🛟 Entry Agent skipped its tool - safety-net save applied")

    elapsed = time.time() - progress["t0"]
    status.update(label=f"✅ All 3 agents finished in {elapsed:.1f}s", state="complete", expanded=False)
    usage = getattr(result, "token_usage", None)
    return {
        "query": query, "age": age, "a1": a1, "a2": a2, "saved_to": saved_to, "elapsed": elapsed,
        "tokens": getattr(usage, "total_tokens", None),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "llm_calls": getattr(usage, "successful_requests", None),
        "time": datetime.now().strftime("%H:%M:%S"),
        "run_id": datetime.now().strftime("%Y%m%d%H%M%S%f"), "cached": False,
    }


@st.cache_resource
def answer_cache() -> dict:
    return {}


def render_result(res: dict) -> None:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("#### 🧠 Answer 1 · WHO knowledge")
            st.caption("From the Assistant's WHO (who.int) knowledge base")
            st.markdown(res["a1"])
    with c2:
        with st.container(border=True):
            st.markdown("#### 🌐 Answer 2 · Trusted web sources")
            st.caption("Fresh results from the web")
            st.markdown(res["a2"])
            srcs = extract_sources(res["a2"])
            if srcs:
                chips = "".join(f"<a href='{u}' target='_blank'>🔗 {domain(u)}</a>" for u in srcs)
                st.markdown(f"<div class='src'>{chips}</div>", unsafe_allow_html=True)

    agree = agreement_score(res["a1"], res["a2"])
    if agree < 20:
        st.warning("🤔 The two answers differ quite a bit - worth checking with your paediatrician.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("⏱ Time", f"{res['elapsed']:.1f}s")
    if res.get("cached"):
        m2.metric("🔢 Tokens", "0", delta="⚡ cached", delta_color="off")
    else:
        m2.metric("🔢 Tokens", f"{res['tokens']:,}" if res.get("tokens") else "—",
                  help=f"Prompt: {res.get('prompt_tokens') or '?'} · Completion: "
                       f"{res.get('completion_tokens') or '?'} · LLM calls: {res.get('llm_calls') or '?'}")
    m3.metric("🤝 Agreement", f"{agree}%", help="How similar the two answers are. Low = double-check.")
    m4.metric("🔗 Web sources", len(extract_sources(res["a2"])))
    st.caption("💗 ParentPal shares general information, not medical advice. "
               "Always follow your paediatrician's guidance.")
    note = "already in" if res.get("cached") else "saved to"
    st.success(f"📔 Entry Agent: {note} your baby journal `{res['saved_to']}`")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="👶", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("results", [])
    st.session_state.setdefault("query", "")

    with st.sidebar:
        st.header("👶 About your child")
        child_name = st.text_input("Child's name (optional)", placeholder="e.g. Aarav")
        age = st.selectbox("Child's age", CHILD_AGES, index=2)
        country = st.selectbox("Country", list(EMERGENCY_NUMBERS), index=0,
                               help="Used for local guidance and emergency numbers.")

        st.header("⚙️ Settings")
        for key in ("OPENAI_API_KEY", "SERPER_API_KEY"):
            if key_status(key):
                st.markdown(f"✅ `{key}` loaded")
            else:
                val = st.text_input(f"Paste {key} (this session only)", type="password", key=f"in_{key}")
                if val:
                    os.environ[key] = val.strip()
                    st.rerun()
        model = st.selectbox("LLM", MODELS, index=0)
        temperature = st.slider("Creativity", 0.0, 1.0, 0.3, 0.1)
        n_results = st.slider("Web results to read", 1, 8, 3)
        use_cache = st.toggle("⚡ Reuse answers for repeated questions", value=True)

        st.header("📚 Assistant knowledge (WHO)")
        live_all = st.toggle("Fetch all WHO pages live", value=False,
                             help="who.int homepage is always fetched live. Turn on to fetch every page live.")
        kb = load_knowledge(KB_SOURCES, live_all)
        for src in kb:
            icon = {"live": "🟢", "snapshot": "🔵", "unavailable": "🔴"}[src["mode"]]
            st.markdown(f"{icon} [{src['title']}]({src['url']})  \n"
                        f"<small>{src['mode']}{' · ' + KB_SNAPSHOT_DATE if src['mode'] == 'snapshot' else ''}"
                        f"</small>", unsafe_allow_html=True)
        st.caption("🟢 live from who.int · 🔵 bundled snapshot")
        if st.button("🔄 Refresh WHO pages"):
            fetch_page.clear()
            st.rerun()

    st.markdown(
        f"<div class='hero'><h1>👶 {APP_TITLE}</h1><p>Your 2 a.m. helpdesk for parenting questions - "
        "WHO guidance and trusted web sources side by side, saved to your baby journal.</p></div>",
        unsafe_allow_html=True,
    )
    tab_ask, tab_journal, tab_how = st.tabs(["💬 Ask", "📔 Baby journal", "🧩 How it works"])

    with tab_ask:
        st.caption("Try a common question:")
        for col, q in zip(st.columns(len(SAMPLE_QUERIES)), SAMPLE_QUERIES):
            if col.button(q):
                st.session_state.query = q
        with st.form("ask"):
            query = st.text_area("What's on your mind?", key="query", height=90,
                                 placeholder="e.g. When can my baby start eating eggs?")
            submitted = st.form_submit_button("💗 Ask ParentPal", type="primary")

        if submitted and query.strip():
            flag = red_flag(query)
            missing = [k for k in ("OPENAI_API_KEY", "SERPER_API_KEY") if not key_status(k)]
            if flag:
                st.markdown(
                    f"<div class='sos'><h3>🚨 This may be an emergency</h3>"
                    f"<b>Call {EMERGENCY_NUMBERS[country]} now</b>, or go to the nearest hospital. "
                    "Don't wait for an online answer.<br><br>While you wait: stay with your child, keep "
                    "them safe, and take any medicine or packaging with you.</div>",
                    unsafe_allow_html=True,
                )
            elif missing:
                st.error(f"Missing API key(s): {', '.join(missing)}.")
            else:
                key = "|".join([re.sub(r"\W+", " ", query.lower()).strip(), age, country, model])
                cache = answer_cache()
                try:
                    if use_cache and key in cache:
                        res = {**cache[key], "cached": True, "elapsed": 0.0,
                               "run_id": datetime.now().strftime("%Y%m%d%H%M%S%f")}
                        st.toast("Answered from cache - 0 tokens", icon="⚡")
                    else:
                        kb_text = retrieve(build_chunks(kb), f"{query} {age}")
                        res = run_crew(query.strip(), model, temperature, n_results, age, country, kb_text)
                        cache[key] = res
                        st.toast("Both answers are ready and saved to your journal!", icon="💗")
                    st.session_state.results.insert(0, res)
                except Exception as exc:
                    st.error("The crew couldn't finish. Check your API keys, quota and network.")
                    with st.expander("Technical details"):
                        st.exception(exc)

        if st.session_state.results:
            res = st.session_state.results[0]
            st.markdown(f"### 💬 {res['query']}  \n<small>Child's age: {res['age']}</small>",
                        unsafe_allow_html=True)
            render_result(res)
            for r in st.session_state.results[1:]:
                with st.expander(f"🕑 {r['time']} · {r['query']}"):
                    render_result(r)
        else:
            render_pipeline(st.empty(), ["waiting"] * 3, [""] * 3)
            st.info("Ask anything about feeding, sleep, vaccines, play or common tummy troubles.")

    with tab_journal:
        entries = read_journal()
        c1, c2 = st.columns([2, 1])
        c1.markdown(f"**{len(entries)}** question(s) in your baby journal")
        if entries:
            c2.download_button("🩺 Prepare for my doctor visit", visit_sheet(entries, child_name),
                               file_name=f"doctor_visit_{datetime.now():%Y%m%d}.txt",
                               help="A printable list of your questions with space for the doctor's answers.")
        search = st.text_input("🔎 Search your journal", placeholder="e.g. sleep, vaccine, fever")
        for e in [e for e in entries if search.lower() in (e["query"] + e["a1"] + e["a2"]).lower()][:50]:
            with st.expander(f"🕑 {e['time']} · {e['age']} · {e['query'][:80]}"):
                a, b = st.columns(2)
                a.markdown("**🧠 WHO knowledge**\n\n" + e["a1"])
                b.markdown("**🌐 Web sources**\n\n" + e["a2"])
        if not entries:
            st.info("Your journal is empty - every question you ask is saved here automatically.")

    with tab_how:
        st.markdown(
            """
```
 Parent's question ──▶ 🚨 red-flag check (no AI) ──▶ emergency card if needed
        │
        ▼
 ┌──────────────────┐  answer 1  ┌───────────────────────┐  answer 2
 │ 1. Assistant     │──────────▶ │ 2. Web Search Asst.   │──────────┐
 │ (WHO knowledge)  │            │ (trusted web sources) │          │
 └──────────────────┘            └───────────────────────┘          ▼
                              Task(context=[...]) ──▶ ┌──────────────────────┐
                                                      │ 3. Entry Agent       │
                                                      │ (Save Journal Entry) │
                                                      └─────────┬────────────┘
                                  baby_journal.txt ◀────────────┤
                                  side-by-side answers ◀────────┘
```
- **Knowledge** - the Assistant answers from WHO pages: who.int (latest news, fetched live) plus WHO guidance on feeding, vaccines, sleep and activity, and diarrhoea. Only the sections that match the question are sent to the model.
- **Safety first** - emergency phrases are caught in plain Python before any AI runs, and every answer reminds parents this is not medical advice.
- **Age aware** - answers adapt to the child's age chosen in the sidebar.
- **Doctor visit sheet** - the journal becomes a printable question list (no AI, 0 tokens).
- **Token-efficient** - compact cached web search, independent web task, by-reference saving, `max_tokens` caps and an answer cache.
- **Secure** - API keys come only from environment variables.
"""
        )


if __name__ == "__main__":
    main()
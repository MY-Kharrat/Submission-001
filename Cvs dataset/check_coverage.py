"""
Coverage check: does the synthetic knowledge base actually fit our tenders?

For every tender in detection/data/simulated_feed/ this prints the best-matching
CVs and projects and exits non-zero when an in-scope tender has fewer than
MIN_STRONG_CVS strong CVs or MIN_STRONG_PROJECTS strong projects.

Why keyword overlap and not embeddings: the rag package (and sentence-transformers)
is not on this branch, and this check must run anywhere in milliseconds. It
mirrors what the pipeline really sees instead:
- records are rendered to text the same way rag/structured_chunking.py builds
  its chunks, so only fields that get embedded can produce a match;
- tokens use the n8n requirements-matrix tokenizer ([a-z0-9+#.]{3,}).

A record is a *strong* match when it names at least one of the tender's core
technologies/capabilities (NAMED_TERMS minus BROAD_TERMS) AND covers at least half of the tender's requirement
lines with a distinctive token. The first condition stops generic words
("data", "platform") from counting as a match -- exactly the failure the old
dataset had. The second stops a single shared tool from counting, which is what
the deliberate distractors test.

tender-014 (catering and facilities) is a negative control: OliveSoft has no
credible match, so ANY strong match there is a failure. That is the brief's
"missing internal references" edge case, and faking coverage would hide it.

Usage (from the repo root or this folder):
    python "Cvs dataset/check_coverage.py"
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from detection.normalize import tag_sector  # noqa: E402  (needs REPO_ROOT on sys.path)

FEED_DIR = REPO_ROOT / "detection" / "data" / "simulated_feed"
CV_PATH = HERE / "fake_cvs.json"
PROJECT_PATH = HERE / "fake_projects.json"

MIN_STRONG_CVS = 3
MIN_STRONG_PROJECTS = 2
LINE_COVERAGE = 0.5
NEGATIVE_CONTROLS = {"tender-014"}
# Duplicate ingest fixtures of one tender; reported together, checked once.
DUPLICATES = {"tender-015": "tender-004"}
VERBATIM_WORDS = 8
TOP_N = 5

CV_KEYS = {"cv_id", "name", "role", "years_experience", "skills", "past_projects", "certifications"}
PAST_PROJECT_KEYS = {"project_name", "client_sector", "description", "tech_stack"}
PROJECT_KEYS = {"project_id", "client_sector", "summary", "tech_stack", "outcome"}

# Tools and capabilities a tender can name. Matched on word boundaries, so a
# record earns the term only when it states it, not a fragment of it.
NAMED_TERMS = [
    "mulesoft", "successfactors", "master data", "payroll", "talend", "ssis", "etl",
    "power bi", "semantic model", "row-level security", "kpi", "cegid", "bigquery", "gcp",
    "identity resolution", "consent", "kafka", "databricks", "lakehouse", "salesforce",
    "service cloud", "omnichannel", "omni-channel", "crm analytics", "machine learning",
    "forecasting", "erp", "retrieval augmented generation", "citations", "computer vision",
    "edge inference", "similarity detection", "crawler", "snowflake", "lineage",
    "de-identification", "telematics", "api integration", "offline",
]

# Real signal, but too broad to prove fit alone: "offline" shows up in mobile,
# vision and CRM work alike, and "salesforce" alone does not mean Service Cloud.
# An earlier version of this check let one of these carry a match, and it
# rated CRM architects as strong fits for a textile computer-vision tender.
BROAD_TERMS = {"payroll", "etl", "kpi", "gcp", "consent", "salesforce", "erp", "citations", "offline"}

STOPWORDS = set("""
the and for with from into over across per not any are all its our their this that
within against between onto each must will has have been which than they them there
what when where who whom whose rather being more most such only also able other
we us you your who about after before during under upon via would should could
""".split())

# Present in nearly every tender and record; covering a line with one of these
# proves nothing about fit.
GENERIC = set("""
data system systems platform integration new layer internal existing management
based services service application applications project solution solutions support
required require requires requirement supplier team teams full specific current
""".split())

TOKEN_RE = re.compile(r"[a-z0-9+#.]{3,}")


def tokens(text):
    return {t.strip(".") for t in TOKEN_RE.findall(text.lower())} - STOPWORDS - {""}


def named_terms(text):
    low = text.lower()
    return {t for t in NAMED_TERMS if re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", low)}


def requirement_lines(tender):
    if tender.get("requirements"):
        return list(tender["requirements"])
    # tender-013 ships without a requirements list; its obligations are the
    # raw_text sentences that state one.
    sentences = re.split(r"(?<=\.)\s+", tender["raw_text"])
    return [s for s in sentences if re.search(r"\b(require|required|must|expect|deploy\w*)\b", s, re.I)]


def cv_text(cv):
    # Same fields rag/structured_chunking.chunk_cv_record puts into its chunks.
    parts = [cv["role"], *cv["skills"], *cv["certifications"]]
    for p in cv["past_projects"]:
        parts += [p["project_name"], p["client_sector"], p["description"], *p["tech_stack"]]
    return " ".join(parts)


def project_text(project):
    # Same fields rag/structured_chunking.chunk_project_record embeds.
    return " ".join([project["summary"], project["client_sector"], *project["tech_stack"], project["outcome"]])


def score(tender, text):
    """Return (is_strong, shared named terms, covered lines, total lines)."""
    lines = tender["_lines"]
    record_tokens = tokens(text)
    shared = tender["_terms"] & named_terms(text)
    covered = sum(1 for line in lines if (tokens(line) - GENERIC) & record_tokens)
    strong = bool(shared - BROAD_TERMS) and covered >= LINE_COVERAGE * len(lines)
    return strong, shared, covered, len(lines)


def validate_schema(cvs, projects):
    problems = []
    if not isinstance(cvs, list) or not isinstance(projects, list):
        return ["top-level JSON must be a list in both files"]
    for cv in cvs:
        if set(cv) != CV_KEYS:
            problems.append(f"{cv.get('cv_id')}: keys {sorted(set(cv) ^ CV_KEYS)} differ")
        for p in cv.get("past_projects", []):
            if set(p) != PAST_PROJECT_KEYS:
                problems.append(f"{cv.get('cv_id')} past project: keys {sorted(set(p) ^ PAST_PROJECT_KEYS)} differ")
    for pr in projects:
        if set(pr) != PROJECT_KEYS:
            problems.append(f"{pr.get('project_id')}: keys {sorted(set(pr) ^ PROJECT_KEYS)} differ")
    return problems


def verbatim_copies(tenders, texts):
    """Tender titles or any VERBATIM_WORDS-word run copied into the data."""
    corpus = " ".join(re.sub(r"\s+", " ", t.lower()) for t in texts)
    hits = []
    for t in tenders:
        if t["title"].lower() in corpus:
            hits.append(f"{t['id']}: title copied")
        words = re.findall(r"[a-z0-9'-]+", " ".join([t["raw_text"], *t["_lines"]]).lower())
        flat = " ".join(re.findall(r"[a-z0-9'-]+", corpus))
        for i in range(len(words) - VERBATIM_WORDS + 1):
            run = " ".join(words[i:i + VERBATIM_WORDS])
            if run in flat:
                hits.append(f"{t['id']}: '{run}'")
                break
    return hits


def main():
    tenders = []
    for path in sorted(FEED_DIR.glob("*.json")):
        t = json.loads(path.read_text(encoding="utf-8"))
        t["_lines"] = requirement_lines(t)
        t["_terms"] = named_terms(" ".join([t["title"], t["raw_text"], *t["_lines"]]))
        t["_line"] = tag_sector(t["raw_text"], t["title"])
        tenders.append(t)

    cvs = json.loads(CV_PATH.read_text(encoding="utf-8"))
    projects = json.loads(PROJECT_PATH.read_text(encoding="utf-8"))
    failures = [f"schema: {p}" for p in validate_schema(cvs, projects)]
    failures += [f"verbatim: {h}" for h in verbatim_copies(
        tenders, [cv_text(c) for c in cvs] + [project_text(p) for p in projects])]

    print(f"{len(cvs)} CVs, {len(projects)} projects, {len(tenders)} tenders\n")
    print(f"{'tender':<11} {'service line':<21} {'strong CV':>9} {'strong PRJ':>10}  status")
    rows = []
    for t in tenders:
        cv_scores = sorted(((score(t, cv_text(c)), c["cv_id"]) for c in cvs),
                           key=lambda x: (x[0][0], len(x[0][1]), x[0][2]), reverse=True)
        pr_scores = sorted(((score(t, project_text(p)), p["project_id"]) for p in projects),
                           key=lambda x: (x[0][0], len(x[0][1]), x[0][2]), reverse=True)
        strong_cvs = [cid for s, cid in cv_scores if s[0]]
        strong_prj = [pid for s, pid in pr_scores if s[0]]

        if t["id"] in NEGATIVE_CONTROLS:
            ok = not strong_cvs and not strong_prj
            status = "ok (negative control)" if ok else "FAIL: out-of-scope tender matched"
        elif t["id"] in DUPLICATES:
            ok = True
            status = f"duplicate of {DUPLICATES[t['id']]}"
        else:
            ok = len(strong_cvs) >= MIN_STRONG_CVS and len(strong_prj) >= MIN_STRONG_PROJECTS
            status = "ok" if ok else f"FAIL: need {MIN_STRONG_CVS} CV / {MIN_STRONG_PROJECTS} PRJ"
        if not ok:
            failures.append(f"{t['id']}: {status}")
        print(f"{t['id']:<11} {t['_line']:<21} {len(strong_cvs):>9} {len(strong_prj):>10}  {status}")
        rows.append((t, cv_scores, pr_scores))

    for t, cv_scores, pr_scores in rows:
        print(f"\n## {t['id']} - {t['title']} [{t['_line']}]")
        print(f"   named terms: {', '.join(sorted(t['_terms'])) or '(none)'}")
        for label, scored in (("CV ", cv_scores), ("PRJ", pr_scores)):
            for (strong, shared, covered, total), rid in scored[:TOP_N]:
                mark = "strong " if strong else ("partial" if shared or covered else "none   ")
                print(f"   {label} {rid} {mark} lines {covered}/{total} terms: {', '.join(sorted(shared)) or '-'}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll tenders meet the coverage target; the negative control stays unmatched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

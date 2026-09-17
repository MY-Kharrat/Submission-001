"""
Turns structured JSON records (CVs, past projects) into clean, natural
language chunks suitable for embedding. Raw JSON embeds poorly -- this
converts each record into readable sentences instead.
"""


def chunk_cv_record(record: dict) -> list[dict]:
    """
    Given one CV record like:
    {
        "cv_id": "CV-0001", "name": "Employee_001", "role": "Solutions Architect",
        "years_experience": 2, "skills": [...], "past_projects": [...], "certifications": [...]
    }
    Returns a list of {"content": ..., "record_id": ..., "record_meta": {...}} dicts --
    one summary chunk plus one chunk per past project.
    """
    chunks = []
    name = record["name"]
    role = record["role"]
    years = record["years_experience"]
    skills = record.get("skills", [])
    certs = record.get("certifications", [])

    # 1. Profile summary chunk
    summary = f"{name} is a {role} with {years} years of experience. Skills: {', '.join(skills)}."
    if certs:
        summary += f" Certifications: {', '.join(certs)}."
    chunks.append({
        "content": summary,
        "record_id": record["cv_id"],
        "record_meta": {"person": name, "role": role, "skills": skills},
    })

    # 2. One chunk per past project this person worked on
    for proj in record.get("past_projects", []):
        text = (
            f"{name} ({role}) worked on '{proj['project_name']}' in the "
            f"{proj['client_sector']} sector. {proj['description']} "
            f"Tech stack: {', '.join(proj['tech_stack'])}."
        )
        chunks.append({
            "content": text,
            "record_id": record["cv_id"],
            "record_meta": {
                "person": name,
                "role": role,
                "project_name": proj["project_name"],
                "client_sector": proj["client_sector"],
                "tech_stack": proj["tech_stack"],
            },
        })

    return chunks


def chunk_project_record(record: dict) -> list[dict]:
    """
    Given one past-project record like:
    {
        "project_id": "PRJ-0001", "client_sector": "energy",
        "summary": "...", "tech_stack": [...], "outcome": "..."
    }
    Returns a single-item list with one clean chunk.
    """
    text = (
        f"{record['summary']} (client sector: {record['client_sector']}). "
        f"Tech stack: {', '.join(record['tech_stack'])}. Outcome: {record['outcome']}."
    )
    return [{
        "content": text,
        "record_id": record["project_id"],
        "record_meta": {
            "client_sector": record["client_sector"],
            "tech_stack": record["tech_stack"],
        },
    }]
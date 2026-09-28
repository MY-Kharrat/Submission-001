"""Convert validated CV/project/capability records into semantic chunks."""


def _required_text(record: dict, field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _string_list(record: dict, field: str) -> list[str]:
    value = record.get(field, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings")
    return [item.strip() for item in value]


def _optional_text(record: dict, field: str) -> str | None:
    value = record.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string when provided")
    return value.strip()


def _source_references(record: dict) -> list[dict]:
    sources = record.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list")
    normalized: list[dict] = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("each source must be an object")
        file_name = _required_text(source, "file")
        page = source.get("page")
        tier = source.get("tier")
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            raise ValueError("source page must be a positive integer")
        if not isinstance(tier, int) or isinstance(tier, bool) or tier < 1:
            raise ValueError("source tier must be a positive integer")
        normalized.append({"file": file_name, "page": page, "tier": tier})
    return normalized


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
    if not isinstance(record, dict):
        raise TypeError("CV record must be an object")
    chunks = []
    record_id = _required_text(record, "cv_id")
    name = _required_text(record, "name")
    role = _required_text(record, "role")
    years = record.get("years_experience")
    if not isinstance(years, int) or isinstance(years, bool) or years < 0:
        raise ValueError("years_experience must be a non-negative integer")
    skills = _string_list(record, "skills")
    certs = _string_list(record, "certifications")

    # 1. Profile summary chunk
    summary = f"{name} is a {role} with {years} years of experience. Skills: {', '.join(skills)}."
    if certs:
        summary += f" Certifications: {', '.join(certs)}."
    chunks.append({
        "content": summary,
        "record_id": record_id,
        "record_meta": {"person": name, "role": role, "skills": skills},
    })

    # 2. One chunk per past project this person worked on
    projects = record.get("past_projects", [])
    if not isinstance(projects, list):
        raise ValueError("past_projects must be a list")
    for proj in projects:
        if not isinstance(proj, dict):
            raise ValueError("each past project must be an object")
        project_name = _required_text(proj, "project_name")
        client_sector = _required_text(proj, "client_sector")
        description = _required_text(proj, "description")
        tech_stack = _string_list(proj, "tech_stack")
        text = (
            f"{name} ({role}) worked on '{project_name}' in the "
            f"{client_sector} sector. {description} "
            f"Tech stack: {', '.join(tech_stack)}."
        )
        chunks.append({
            "content": text,
            "record_id": record_id,
            "record_meta": {
                "person": name,
                "role": role,
                "project_name": project_name,
                "client_sector": client_sector,
                "tech_stack": tech_stack,
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
    if not isinstance(record, dict):
        raise TypeError("project record must be an object")
    record_id = _required_text(record, "project_id")
    summary = _required_text(record, "summary")
    client_sector = _required_text(record, "client_sector")
    outcome = _required_text(record, "outcome")
    tech_stack = _string_list(record, "tech_stack")
    text = (
        f"{summary} (client sector: {client_sector}). "
        f"Tech stack: {', '.join(tech_stack)}. Outcome: {outcome}."
    )
    return [{
        "content": text,
        "record_id": record_id,
        "record_meta": {
            "client_sector": client_sector,
            "tech_stack": tech_stack,
        },
    }]


def chunk_olivesoft_project_record(record: dict) -> list[dict]:
    """Convert a source-attributed OliveSoft project catalogue record.

    Catalogue entries retain their status (for example ``proposed``), so they
    can influence requirement matching without being presented as completed
    client deliveries.
    """
    if not isinstance(record, dict):
        raise TypeError("OliveSoft project record must be an object")
    record_id = _required_text(record, "id")
    name = _required_text(record, "name")
    relationship = _required_text(record, "relationship")
    edition = _required_text(record, "edition")
    description = _required_text(record, "description")
    outcome = _required_text(record, "outcome").lower()
    confidence = _required_text(record, "confidence").lower()
    if confidence not in {"low", "medium", "high"}:
        raise ValueError("confidence must be low, medium, or high")

    domains = _string_list(record, "domain")
    aliases = _string_list(record, "aliases")
    tasks = _string_list(record, "tasks")
    stack = record.get("tech_stack")
    if not isinstance(stack, dict):
        raise ValueError("tech_stack must be an object")
    technologies = _string_list(stack, "technologies")
    sources = _source_references(record)
    client = _optional_text(record, "client") or "Not specified"
    duration = _optional_text(record, "duration")
    notes = _optional_text(record, "notes")
    is_completed = outcome in {"completed", "delivered", "production"}

    text = (
        f"OliveSoft project initiative '{name}' ({edition}, status: {outcome}, "
        f"relationship: {relationship}). {description} "
        f"Domains: {', '.join(domains)}. Technologies: {', '.join(technologies)}. "
        f"Planned work: {'; '.join(tasks)}. Client context: {client}."
    )
    if aliases:
        text += f" Also known as: {', '.join(aliases)}."
    if duration:
        text += f" Duration: {duration}."
    if notes:
        text += f" Source note: {notes}."

    return [{
        "content": text,
        "record_id": record_id,
        "record_meta": {
            "name": name,
            "project_origin": "olivesoft",
            "project_kind": relationship,
            "evidence_status": outcome,
            "is_completed": is_completed,
            "matching_priority_boost": 0.08,
            "edition": edition,
            "client": client,
            "domains": domains,
            "tech_stack": technologies,
            "tasks": tasks,
            "aliases": aliases,
            "duration": duration,
            "confidence": confidence,
            "source_references": sources,
            "notes": notes,
        },
    }]


def chunk_tool_record(record: dict) -> list[dict]:
    """Convert one OliveSoft capability/tool record into searchable evidence."""
    if not isinstance(record, dict):
        raise TypeError("tool record must be an object")
    record_id = _required_text(record, "tool_id")
    name = _required_text(record, "name")
    category = _required_text(record, "category")
    description = _required_text(record, "description")
    capabilities = _string_list(record, "capabilities")
    technologies = _string_list(record, "technologies")
    use_cases = _string_list(record, "use_cases")
    certifications = _string_list(record, "certifications")
    text = (
        f"OliveSoft capability '{name}' in {category}. {description} "
        f"Capabilities: {', '.join(capabilities)}. "
        f"Technologies: {', '.join(technologies)}. "
        f"Use cases: {', '.join(use_cases)}."
    )
    if certifications:
        text += f" Standards and certifications: {', '.join(certifications)}."
    return [{
        "content": text,
        "record_id": record_id,
        "record_meta": {
            "name": name,
            "category": category,
            "capabilities": capabilities,
            "technologies": technologies,
            "use_cases": use_cases,
            "certifications": certifications,
        },
    }]

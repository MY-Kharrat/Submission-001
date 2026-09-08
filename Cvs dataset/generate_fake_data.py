"""
Fake CV & project portfolio generator for Olivesoft RFP Intelligence — Phase 1.

Generates synthetic, clearly-fake internal knowledge data (CVs + past projects)
matching the schema agreed in the team plan. No real personal data is used.

Usage:
    python generate_fake_data.py --cvs 20 --projects 12 --seed 42

Outputs:
    fake_cvs.json
    fake_projects.json
"""

import argparse
import json
import random

ROLES = [
    "Backend Developer", "Frontend Developer", "Full Stack Developer",
    "Data Engineer", "Data Scientist", "DevOps Engineer", "Cloud Architect",
    "QA Engineer", "Mobile Developer", "AI/ML Engineer", "Solutions Architect",
    "UI/UX Designer", "Project Manager", "Business Analyst",
]

SKILLS_POOL = [
    "Python", "Java", "JavaScript", "TypeScript", "React", "Next.js",
    "Node.js", "FastAPI", "Django", "Spring Boot", "Docker", "Kubernetes",
    "AWS", "Azure", "GCP", "PostgreSQL", "MongoDB", "Redis",
    "LangChain", "LlamaIndex", "Qdrant", "ChromaDB", "Pinecone",
    "n8n", "REST APIs", "GraphQL", "CI/CD", "Terraform", "Tailwind CSS",
    "Flutter", "Swift", "Kotlin", "TensorFlow", "PyTorch", "Scikit-learn",
]

CLIENT_SECTORS = [
    "banking", "healthcare", "public administration", "retail",
    "telecom", "insurance", "logistics", "energy", "education",
    "manufacturing", "real estate",
]

CERTIFICATIONS_POOL = [
    "AWS Certified Solutions Architect", "AWS Certified Developer",
    "Certified Kubernetes Administrator", "PMP", "Scrum Master (CSM)",
    "Google Professional Cloud Architect", "Microsoft Azure Fundamentals",
    None, None, None,  # weight toward "no certification" being common
]

PROJECT_NAME_TEMPLATES = [
    "{sector} customer portal revamp",
    "{sector} internal workflow automation",
    "{sector} data platform migration",
    "{sector} mobile app for field agents",
    "{sector} AI-powered document processing",
    "{sector} legacy system modernization",
    "{sector} real-time analytics dashboard",
    "{sector} API integration hub",
]

OUTCOME_TEMPLATES = [
    "Reduced manual processing time by {pct}%.",
    "Improved system response time by {pct}%.",
    "Delivered on schedule with a {pct}% reduction in reported defects.",
    "Increased platform adoption by {pct}% within the first quarter.",
    "Cut infrastructure costs by {pct}% after migration.",
]


def make_cv(cv_index, rng):
    role = rng.choice(ROLES)
    years = rng.randint(1, 15)
    skills = rng.sample(SKILLS_POOL, k=rng.randint(4, 8))
    n_projects = rng.randint(1, 4)
    past_projects = []
    for _ in range(n_projects):
        sector = rng.choice(CLIENT_SECTORS)
        past_projects.append({
            "project_name": rng.choice(PROJECT_NAME_TEMPLATES).format(
                sector=sector.capitalize()
            ),
            "client_sector": sector,
            "description": (
                f"Contributed as {role.lower()} on a {sector} sector project, "
                f"working with {', '.join(rng.sample(skills, k=min(3, len(skills))))}."
            ),
            "tech_stack": rng.sample(skills, k=min(3, len(skills))),
        })
    cert = rng.choice(CERTIFICATIONS_POOL)
    return {
        "cv_id": f"CV-{cv_index:04d}",
        "name": f"Employee_{cv_index:03d}",
        "role": role,
        "years_experience": years,
        "skills": skills,
        "past_projects": past_projects,
        "certifications": [cert] if cert else [],
    }


def make_project(project_index, rng):
    sector = rng.choice(CLIENT_SECTORS)
    tech_stack = rng.sample(SKILLS_POOL, k=rng.randint(3, 6))
    pct = rng.randint(15, 60)
    return {
        "project_id": f"PRJ-{project_index:04d}",
        "client_sector": sector,
        "summary": rng.choice(PROJECT_NAME_TEMPLATES).format(
            sector=sector.capitalize()
        ),
        "tech_stack": tech_stack,
        "outcome": rng.choice(OUTCOME_TEMPLATES).format(pct=pct),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate fake CV and project data.")
    parser.add_argument("--cvs", type=int, default=20, help="Number of fake CVs to generate")
    parser.add_argument("--projects", type=int, default=12, help="Number of fake projects to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--cv-out", type=str, default="fake_cvs.json", help="Output path for CVs")
    parser.add_argument("--project-out", type=str, default="fake_projects.json", help="Output path for projects")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    cvs = [make_cv(i + 1, rng) for i in range(args.cvs)]
    projects = [make_project(i + 1, rng) for i in range(args.projects)]

    with open(args.cv_out, "w", encoding="utf-8") as f:
        json.dump(cvs, f, indent=2, ensure_ascii=False)

    with open(args.project_out, "w", encoding="utf-8") as f:
        json.dump(projects, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(cvs)} fake CVs -> {args.cv_out}")
    print(f"Generated {len(projects)} fake projects -> {args.project_out}")


if __name__ == "__main__":
    main()

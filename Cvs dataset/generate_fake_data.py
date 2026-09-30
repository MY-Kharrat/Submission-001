"""
Fake CV & project portfolio generator for Olivesoft RFP Intelligence.

Generates synthetic, clearly-fake internal knowledge data (CVs + past projects)
matching the schema agreed in the team plan. No real personal data is used.

Why the vocabulary is tender-aware: the first version organised everything
around the *buyer's industry* (banking, retail...), while tenders are organised
around *the work being bought* -- OliveSoft's five service lines and the exact
tools each tender names. Retrieval over that data could only ever match on
generic words like "data" or "platform", so the RAG benchmark measured noise.
Every record here is built from requirement-level specialisms instead, so each
simulated tender has people and references that genuinely fit it.

Why it is still not trivially easy to match:
- Phrasing is paraphrased, never copied from tender text. Verbatim overlap would
  inflate similarity scores and hide real retrieval weaknesses.
- Distractors are deliberate: right tool in the wrong domain, right domain at the
  wrong seniority, an adjacent tool instead of the named one. A retriever that
  cannot rank these below true matches should fail the benchmark.
- About 30% of CVs are OliveSoft's web/mobile/custom-software bench -- the
  company profile in the challenge brief -- so data/AI tenders must be matched
  against a realistic mix, not a corpus that only contains the right answers.

Output format is unchanged from the original generator: same keys, same types,
same ID scheme (CV-0001, PRJ-0001, Employee_001). The same --seed always gives
byte-identical files, on any OS.

Usage:
    python generate_fake_data.py --cvs 59 --projects 35 --seed 42

Outputs:
    fake_cvs.json
    fake_projects.json
"""

import argparse
import json
import random

# Mirrors shared.schemas.CAPABILITY_TAXONOMY. Duplicated rather than imported so
# this script runs standalone from any working directory; check_coverage.py is
# what verifies the generated data against the real pipeline.
DATA_INTEGRATION = "Data Integration"
AI_DEVELOPMENT = "AI Development"
BI_DASHBOARDING = "BI & Dashboarding"
SALESFORCE = "Salesforce Ecosystem"
DATA_PLATFORM = "Data Platform"

# Buyer industries, modelled on the *kinds* of organisation that issue our
# tenders. Only sector labels -- never an issuer or any real organisation name.
SECTORS = {
    "consumer_goods": "consumer goods",
    "logistics": "logistics",
    "industrial": "industrial manufacturing",
    "retail": "retail",
    "fashion": "fashion retail",
    "insurance": "insurance",
    "mining": "mining and raw materials",
    "justice": "public sector (justice)",
    "interior": "public sector (interior)",
    "statistics": "public statistics",
    "health": "healthcare",
    "telecom": "telecom",
    "energy": "energy utilities",
    "banking": "banking",
}

# One entry per requirement-level need found in the simulated tender feed.
# Each is guaranteed CVS_PER_SPECIALISM CVs and every listed project variant,
# which is what gives each tender >= 3 strong CVs and >= 2 strong projects.
SPECIALISMS = [
    {
        "key": "hr_integration",
        "line": DATA_INTEGRATION,
        "roles": [("Senior Integration Architect", 8, 15), ("Integration Developer", 4, 9),
                  ("MuleSoft Developer", 3, 8)],
        "skills": ["MuleSoft", "SAP SuccessFactors", "Master Data Management", "API-led connectivity"],
        "extra": ["DataWeave", "REST APIs", "Anypoint Platform", "Java", "OAuth 2.0"],
        "certs": ["MuleSoft Certified Integration Architect", "MuleSoft Certified Developer"],
        "sectors": ["consumer_goods", "industrial", "logistics"],
        "cv_desc": [
            "Built MuleSoft flows linking SuccessFactors to regional payroll engines, with master "
            "data rules that stopped duplicate employee records appearing across countries.",
            "Designed versioned API contracts and a handover pack so the client's IT team could "
            "run the integration estate, including country-specific statutory payroll exports.",
        ],
        "projects": [
            ("Multi-country HR and payroll integration hub for {a_sector} group",
             ["MuleSoft", "SAP SuccessFactors", "Master Data Management", "DataWeave"],
             "Single employee golden record across {n} payroll providers; duplicate hires fell by {pct}%."),
            ("Employee master data consolidation with versioned APIs for {a_sector} company",
             ["MuleSoft", "Anypoint Platform", "REST APIs", "Master Data Management"],
             "Statutory payroll feeds validated per country; internal IT took over support after handover documentation was signed off."),
        ],
    },
    {
        "key": "etl_migration",
        "line": DATA_INTEGRATION,
        "roles": [("Senior ETL Engineer", 7, 14), ("Data Integration Engineer", 3, 8),
                  ("ETL Developer", 2, 6)],
        "skills": ["Talend", "SSIS", "ETL", "SQL Server", "Incremental loading"],
        "extra": ["Python", "Azure Data Factory", "Data reconciliation", "Git"],
        "certs": ["Talend Data Integration Certified Developer", "Microsoft Certified: Azure Data Engineer Associate"],
        "sectors": ["logistics", "consumer_goods", "insurance"],
        "cv_desc": [
            "Migrated legacy SSIS packages to Talend jobs, adding incremental loads that survive a "
            "partial failure without reprocessing the full history.",
            "Wrote reconciliation checks comparing legacy and migrated extracts so drift in "
            "recruitment and applicant tracking data was caught before go-live.",
        ],
        "projects": [
            ("SSIS to Talend pipeline migration for {a_sector} operator",
             ["Talend", "SSIS", "SQL Server", "ETL"],
             "{n} legacy packages retired; incremental loads recover from partial failures with no manual reruns."),
            ("Recruitment and applicant tracking ETL modernisation for {a_sector} group",
             ["Talend", "ETL", "Python", "Data reconciliation"],
             "Automated reconciliation report flags extract drift within one day; load window cut by {pct}%."),
        ],
    },
    {
        "key": "hr_bi",
        "line": BI_DASHBOARDING,
        "roles": [("Senior BI Consultant", 7, 14), ("Power BI Developer", 3, 8),
                  ("BI Solution Architect", 9, 15)],
        "skills": ["Power BI", "DAX", "Semantic model design", "Row-level security", "KPI governance"],
        "extra": ["SQL", "Power Query", "Tabular Editor", "Azure Synapse"],
        "certs": ["Microsoft Certified: Power BI Data Analyst Associate", "Microsoft Certified: Fabric Analytics Engineer Associate"],
        "sectors": ["industrial", "consumer_goods", "health"],
        "cv_desc": [
            "Replaced divisional spreadsheet packs with one governed Power BI semantic model whose "
            "KPI definitions were agreed with the HR leadership team.",
            "Implemented row-level security by division and multi-year historical trend views for "
            "workforce, absence and attrition reporting.",
        ],
        "projects": [
            ("Governed workforce analytics in Power BI for {a_sector} organisation",
             ["Power BI", "DAX", "Semantic model design", "Row-level security"],
             "{n} divisional reporting packs replaced; HR managers see only their own division."),
            ("HR KPI semantic model and executive dashboards for {a_sector} organisation",
             ["Power BI", "KPI governance", "SQL", "Power Query"],
             "Three years of historical workforce data comparable on one model; report preparation time down {pct}%."),
        ],
    },
    {
        "key": "customer360",
        "line": DATA_PLATFORM,
        "roles": [("Senior Data Engineer", 7, 13), ("Customer Data Architect", 9, 15),
                  ("Data Engineer", 3, 7)],
        "skills": ["BigQuery", "GCP", "Identity resolution", "Consent management", "dbt"],
        "extra": ["Cegid ERP", "Pub/Sub", "Python", "Looker Studio", "Data quality"],
        "certs": ["Google Professional Data Engineer", "Google Professional Cloud Architect"],
        "sectors": ["retail", "fashion", "consumer_goods"],
        "cv_desc": [
            "Unified ERP, e-commerce, marketing and support customer records in a BigQuery landing "
            "zone, matching people on email and loyalty id with a postcode fallback.",
            "Built consent and opt-out propagation so downstream marketing tools honoured a customer "
            "withdrawal the same day.",
        ],
        "projects": [
            ("Customer single view on BigQuery for {a_sector} chain",
             ["BigQuery", "GCP", "Identity resolution", "Cegid ERP"],
             "{pct}% of customers resolved to one profile across ERP, commerce and support systems."),
            ("Consent-aware customer data platform on GCP for {a_sector} brand",
             ["BigQuery", "GCP", "Consent management", "Pub/Sub"],
             "Opt-outs reach every downstream system in under {n} hours; marketing complaints fell sharply."),
        ],
    },
    {
        "key": "streaming",
        "line": DATA_PLATFORM,
        "roles": [("Senior Streaming Engineer", 7, 14), ("Data Platform Engineer", 4, 9),
                  ("Lakehouse Architect", 9, 15)],
        "skills": ["Kafka", "Databricks", "Lakehouse", "Delta Lake", "Late-arriving event handling"],
        "extra": ["Spark Structured Streaming", "Python", "Grafana", "Data observability"],
        "certs": ["Databricks Certified Data Engineer Professional", "Confluent Certified Developer for Apache Kafka"],
        "sectors": ["fashion", "retail", "logistics"],
        "cv_desc": [
            "Streamed order and warehouse events through Kafka into a Databricks lakehouse, with "
            "watermarking so late events corrected earlier aggregates instead of being dropped.",
            "Modelled multi-currency sales and returns, and added pipeline health monitoring with "
            "alerting on lag and schema changes.",
        ],
        "projects": [
            ("Order event streaming into a Databricks lakehouse for {a_sector} business",
             ["Kafka", "Databricks", "Delta Lake", "Spark Structured Streaming"],
             "Order visibility moved from overnight to near real time; late events reconciled automatically."),
            ("Multi-market sales and returns lakehouse model for {a_sector} brand",
             ["Databricks", "Lakehouse", "Kafka", "Data observability"],
             "Currency-normalised sales across {n} markets; pipeline incidents detected {pct}% faster."),
        ],
    },
    {
        "key": "service_cloud",
        "line": SALESFORCE,
        "roles": [("Salesforce Technical Architect", 9, 15), ("Salesforce Developer", 3, 8),
                  ("Salesforce Consultant", 4, 10)],
        "skills": ["Salesforce Service Cloud", "Omni-Channel routing", "Apex", "Case migration"],
        "extra": ["Salesforce Flow", "Data Loader", "Change management", "Lightning Web Components"],
        "certs": ["Salesforce Certified Service Cloud Consultant", "Salesforce Certified Application Architect"],
        "sectors": ["insurance", "banking", "energy"],
        "cv_desc": [
            "Rolled out Service Cloud with a single omnichannel queue for phone, email and chat, "
            "migrating more than a decade of historical cases.",
            "Integrated case records with a policy administration system and ran a parallel-run "
            "cutover plus adviser training and change management.",
        ],
        "projects": [
            ("Service Cloud contact centre rollout for {a_sector} provider",
             ["Salesforce Service Cloud", "Omni-Channel routing", "Case migration", "Data Loader"],
             "{n} years of case history migrated; average handling time down {pct}%."),
            ("Policy-aware case management on Salesforce for {a_sector} group",
             ["Salesforce Service Cloud", "Apex", "Salesforce Flow", "Change management"],
             "Case and policy data kept consistent; phased cutover completed with no service outage."),
        ],
    },
    {
        "key": "forecasting",
        "line": AI_DEVELOPMENT,
        "roles": [("Senior Data Scientist", 7, 14), ("Machine Learning Engineer", 3, 8),
                  ("Forecasting Lead", 9, 15)],
        "skills": ["Machine learning", "Demand forecasting", "Explainable AI", "Prediction intervals"],
        "extra": ["Python", "LightGBM", "SHAP", "MLflow", "ERP data integration"],
        "certs": ["Google Professional Machine Learning Engineer", "AWS Certified Machine Learning - Specialty"],
        "sectors": ["mining", "industrial", "consumer_goods"],
        "cv_desc": [
            "Built machine learning demand forecasts per product and customer from years of ERP order "
            "history, exposing prediction intervals rather than single numbers.",
            "Added SHAP explanations so planners could challenge a forecast, and fed their manual "
            "overrides back as training signal.",
        ],
        "projects": [
            ("Customer-level demand forecasting for {a_sector} supplier",
             ["Machine learning", "Demand forecasting", "LightGBM", "ERP data integration"],
             "Stockouts on high-value lines down {pct}%; forecasts published with confidence bands."),
            ("Explainable forecasting with planner feedback loop for {a_sector} company",
             ["Machine learning", "Explainable AI", "SHAP", "MLflow"],
             "Planner overrides captured and retrained on monthly; forecast error improved by {pct}%."),
        ],
    },
    {
        "key": "rag",
        "line": AI_DEVELOPMENT,
        "roles": [("Senior AI Engineer", 6, 12), ("LLM Application Engineer", 3, 7),
                  ("AI Solutions Architect", 9, 15)],
        "skills": ["Retrieval augmented generation", "LLM evaluation", "Source citations", "pgvector"],
        "extra": ["LangChain", "LlamaIndex", "Python", "FastAPI", "Prompt injection defense"],
        "certs": ["Microsoft Certified: Azure AI Engineer Associate"],
        "sectors": ["retail", "insurance", "telecom"],
        "cv_desc": [
            "Shipped a retrieval augmented generation assistant over an approved document corpus "
            "that cites document and page, and refuses with a human handover when evidence is missing.",
            "Built an evaluation harness replaying hundreds of historical customer questions against "
            "an agreed accuracy threshold before each release.",
        ],
        "projects": [
            ("Grounded customer support assistant for {a_sector} business",
             ["Retrieval augmented generation", "LlamaIndex", "pgvector", "Source citations"],
             "Every answer cites its source page; unsupported questions routed to agents instead of guessed."),
            ("LLM answer quality evaluation programme for {a_sector} service desk",
             ["LLM evaluation", "Retrieval augmented generation", "LangChain", "Python"],
             "Accuracy measured on {n} historical queries per release; regressions blocked before deployment."),
        ],
    },
    {
        "key": "vision",
        "line": AI_DEVELOPMENT,
        "roles": [("Computer Vision Engineer", 4, 10), ("Senior ML Engineer (Vision)", 8, 14),
                  ("Edge AI Engineer", 3, 8)],
        "skills": ["Computer vision", "Defect detection", "Edge inference", "PyTorch"],
        "extra": ["ONNX Runtime", "NVIDIA Jetson", "OpenCV", "Label adjudication"],
        "certs": ["NVIDIA DLI: Building AI-Based Defect Detection"],
        "sectors": ["fashion", "industrial"],
        "cv_desc": [
            "Trained computer vision models for fabric and garment defect classes, tuned to favour "
            "catching safety-relevant defects over avoiding false alarms.",
            "Deployed edge inference that keeps working offline on the factory floor and syncs "
            "results when connectivity returns; ran adjudication to clean noisy inspector labels.",
        ],
        "projects": [
            ("Edge visual inspection for textile quality control at {a_sector} plant",
             ["Computer vision", "Edge inference", "PyTorch", "NVIDIA Jetson"],
             "{n} defect classes detected on the line; runs offline with opportunistic sync."),
            ("Inspection label cleanup and defect model training for {a_sector} producer",
             ["Computer vision", "Defect detection", "Label adjudication", "OpenCV"],
             "Ground truth rebuilt from conflicting inspector decisions; missed safety defects down {pct}%."),
        ],
    },
    {
        "key": "similarity",
        "line": AI_DEVELOPMENT,
        "roles": [("Senior Search & ML Engineer", 7, 13), ("Information Retrieval Engineer", 4, 9),
                  ("Data Scientist (Similarity Search)", 3, 8)],
        "skills": ["Similarity detection", "Image embeddings", "Text similarity", "Web crawler"],
        "extra": ["Elasticsearch", "FAISS", "Python", "Audit logging"],
        "certs": ["Elastic Certified Engineer"],
        "sectors": ["justice", "interior"],
        "cv_desc": [
            "Built similarity detection over logo images and word marks, returning ranked candidates "
            "with the matched evidence shown rather than an opaque score.",
            "Handled national and international classification codes and kept an auditable decision "
            "trail that could withstand an appeal; wrote the prior-art web crawler.",
        ],
        "projects": [
            ("Trademark image and word-mark similarity search for {a_sector} registry",
             ["Similarity detection", "Image embeddings", "Text similarity", "FAISS"],
             "Examiners review ranked candidates with the matched basis visible; review time down {pct}%."),
            ("Prior-art crawler with auditable matching for {a_sector} authority",
             ["Web crawler", "Similarity detection", "Elasticsearch", "Audit logging"],
             "Every match decision reproducible for external appeal; class code filters across {n} classes."),
        ],
    },
    {
        "key": "crm_analytics",
        "line": SALESFORCE,
        "roles": [("Salesforce CRM Analytics Consultant", 5, 11), ("Senior Salesforce Architect", 9, 15),
                  ("Salesforce Developer", 3, 7)],
        "skills": ["Salesforce CRM Analytics", "Salesforce Field Service", "Offline sync", "Data residency"],
        "extra": ["Apex", "SAQL", "Salesforce Shield", "Case management"],
        "certs": ["Salesforce Certified CRM Analytics and Einstein Discovery Consultant", "Salesforce Certified Platform Developer I"],
        "sectors": ["interior", "justice"],
        "cv_desc": [
            "Built CRM Analytics dashboards on an existing Salesforce org giving central oversight of "
            "case volumes, resolution times and backlog per office each month.",
            "Delivered offline working for branch offices with reconciliation on reconnect, and kept "
            "citizen records inside government-hosted infrastructure for data residency.",
        ],
        "projects": [
            ("Citizen case analytics on Salesforce for {a_sector} ministry",
             ["Salesforce CRM Analytics", "SAQL", "Case management", "Apex"],
             "Monthly backlog and resolution reporting across {n} offices generated automatically."),
            ("Offline-capable case handling and oversight dashboards for {a_sector} agency",
             ["Salesforce CRM Analytics", "Salesforce Field Service", "Offline sync", "Data residency"],
             "Offices keep working through outages and reconcile on reconnect; records never leave sovereign hosting."),
        ],
    },
    {
        "key": "stats_platform",
        "line": DATA_PLATFORM,
        "roles": [("Senior Data Platform Engineer", 7, 14), ("Analytics Engineer", 3, 8),
                  ("Data Governance Lead", 9, 15)],
        "skills": ["BigQuery", "Snowflake", "Data lineage", "De-identification", "Schema validation"],
        "extra": ["dbt", "GCP", "Great Expectations", "Open data publishing"],
        "certs": ["SnowPro Core Certification", "Google Professional Data Engineer"],
        "sectors": ["statistics", "interior", "health"],
        "cv_desc": [
            "Built a BigQuery warehouse for validated statistical series and Snowflake models that "
            "trace every published figure back to the unit that submitted it.",
            "Enforced submission specifications that reject malformed files with precise error "
            "messages, and ran a documented de-identification review before open data release.",
        ],
        "projects": [
            ("Official statistics warehouse with end-to-end lineage for {a_sector} body",
             ["BigQuery", "Snowflake", "Data lineage", "dbt"],
             "Each published series traceable to its submitting unit; {pct}% fewer correction notices."),
            ("Validated submissions and open data release pipeline for {a_sector} body",
             ["Schema validation", "De-identification", "Great Expectations", "GCP"],
             "Non-conforming files rejected with specific errors; every release passes a signed disclosure review."),
        ],
    },
    {
        "key": "telematics_mobile",
        "line": DATA_INTEGRATION,
        "roles": [("Senior Mobile Developer", 6, 12), ("Integration Engineer (IoT)", 4, 9),
                  ("Mobile Tech Lead", 8, 14)],
        "skills": ["Telematics integration", "API integration", "Flutter", "Offline-first mobile"],
        "extra": ["Kotlin", "SQLite", "Localisation", "MQTT", "Proof of delivery"],
        "certs": ["Google Associate Android Developer"],
        "sectors": ["mining", "logistics", "energy"],
        "cv_desc": [
            "Connected fleet telematics feeds to maintenance and fuel systems, with a defined fallback "
            "when the telematics provider goes offline for an extended period.",
            "Built an offline-first driver app recording delivery confirmations and digital logbook "
            "entries, localised into the drivers' own language and rolled out without disrupting dispatch.",
        ],
        "projects": [
            ("Fleet telematics integration with maintenance and fuel systems for {a_sector} operator",
             ["Telematics integration", "API integration", "MQTT", "Kotlin"],
             "Vehicle events reach the systems that act on them; {n} vehicles connected with no dispatch downtime."),
            ("Offline-first driver app fed by telematics data for {a_sector} company",
             ["Flutter", "Offline-first mobile", "API integration", "Localisation"],
             "Delivery proof and logbooks captured in poor connectivity across {n} depots; paper retired within {pct} days."),
        ],
    },
]

CVS_PER_SPECIALISM = 3

# The specbook describes OliveSoft as enterprise digital solutions, web/mobile
# engineering and custom software. These profiles represent that bench: they
# are real capacity, and for data/AI tenders they are the "plausible but wrong"
# candidates a good retriever must rank lower.
WEB_MOBILE_BENCH = [
    ("Frontend Developer", ["React", "Next.js", "TypeScript", "Tailwind CSS"], "customer_portal"),
    ("Frontend Developer", ["React", "TypeScript", "Storybook", "Accessibility (WCAG)"], "customer_portal"),
    ("Full Stack Developer", ["Next.js", "Node.js", "PostgreSQL", "TypeScript"], "b2b_ordering"),
    ("Full Stack Developer", ["React", "FastAPI", "PostgreSQL", "Docker"], "workflow_app"),
    ("Mobile Developer", ["Flutter", "Dart", "Firebase", "REST APIs"], "field_inspection"),
    ("Mobile Developer", ["Swift", "Kotlin", "REST APIs", "CI/CD"], "mobile_banking"),
    ("Backend Developer", ["Java", "Spring Boot", "PostgreSQL", "REST APIs"], "b2b_ordering"),
    ("Backend Developer", ["Node.js", "GraphQL", "MongoDB", "Redis"], "workflow_app"),
    ("DevOps Engineer", ["Docker", "Kubernetes", "Terraform", "CI/CD"], "workflow_app"),
    ("Cloud Architect", ["AWS", "Terraform", "Kubernetes", "Serverless"], "customer_portal"),
    ("QA Engineer", ["Playwright", "Cypress", "API testing", "CI/CD"], "customer_portal"),
    ("UI/UX Designer", ["Figma", "Design systems", "User research", "Prototyping"], "mobile_banking"),
    ("Project Manager", ["Scrum", "Stakeholder management", "Delivery planning", "Jira"], "workflow_app"),
    ("Business Analyst", ["Requirements analysis", "BPMN", "User stories", "Workshops"], "b2b_ordering"),
]

BENCH_PROJECTS = {
    "customer_portal": ("Customer self-service web portal for {a_sector} provider",
                        ["React", "Next.js", "TypeScript", "PostgreSQL"],
                        "Self-service adoption up {pct}%; call volumes fell in the first quarter."),
    "b2b_ordering": ("B2B ordering web shop for {a_sector} distributor",
                     ["Next.js", "Spring Boot", "PostgreSQL", "REST APIs"],
                     "Online share of orders grew to {pct}% within a year."),
    "workflow_app": ("Custom approval workflow application for {a_sector} back office",
                     ["React", "FastAPI", "Docker", "Kubernetes"],
                     "Manual approval steps cut by {pct}%; {n} paper forms retired."),
    "field_inspection": ("Offline field inspection mobile app for {a_sector} network operator",
                         ["Flutter", "Firebase", "SQLite", "REST APIs"],
                         "Inspectors capture findings without signal; reports filed {pct}% faster."),
    "mobile_banking": ("Retail mobile banking app refresh for {a_sector} institution",
                       ["Swift", "Kotlin", "REST APIs", "CI/CD"],
                       "App store rating rose to 4.{n}; release cadence doubled."),
}

# Per bench project, so a banking app never lands at a telecom "institution".
BENCH_SECTORS = {
    "customer_portal": ["insurance", "energy", "telecom"],
    "b2b_ordering": ["consumer_goods", "industrial"],
    "workflow_app": ["banking", "insurance", "health"],
    "field_inspection": ["energy", "telecom"],
    "mobile_banking": ["banking"],
}

# Deliberate near misses, each teaching the retriever something different.
# Kept out of SPECIALISMS so they never count toward a tender's guaranteed quota.
DISTRACTOR_CVS = [
    # Right tool, wrong domain: Kafka experience, but for telecom billing, with
    # no lakehouse, returns or currency work.
    ("Backend Developer", 5, ["Kafka", "Java", "Spring Boot", "PostgreSQL"],
     "Built Kafka consumers for a telecom billing mediation service.", "telecom", "kafka_billing"),
    # Right domain, wrong seniority: HR dashboards, but a junior report builder
    # rather than someone who can govern a semantic model.
    ("Junior BI Analyst", 1, ["Power BI", "Excel", "SQL"],
     "Maintained monthly headcount reports for an HR team under supervision.", "industrial", None),
    # Adjacent tool: classic ETL migration, but Informatica rather than Talend/SSIS.
    ("ETL Developer", 6, ["Informatica PowerCenter", "Oracle", "ETL", "Shell scripting"],
     "Migrated Informatica workflows between Oracle environments.", "banking", "informatica_erp"),
    # Adjacent capability: a scripted FAQ chatbot, without grounding, citations or evaluation.
    ("Conversational AI Developer", 4, ["Dialogflow", "Chatbot", "Node.js", "Intent design"],
     "Built a scripted FAQ chatbot answering delivery-status questions.", "retail", "faq_chatbot"),
    # Generalist spanning lines: Salesforce + MuleSoft, useful but not a specialist in either tender.
    ("Integration Consultant", 10, ["Salesforce Sales Cloud", "MuleSoft", "REST APIs", "Pre-sales"],
     "Connected Sales Cloud opportunities to an ERP through a small MuleSoft layer.", "consumer_goods", None),
    # Generalist: data engineer with some BI, no named tool from any tender.
    ("Data Engineer", 6, ["Python", "Airflow", "PostgreSQL", "Metabase"],
     "Ran nightly batch pipelines and a shared KPI dashboard for an operations team.", "logistics", None),
]

DISTRACTOR_PROJECTS = {
    "kafka_billing": ("Telecom billing mediation on event streams",
                      ["Kafka", "Java", "Spring Boot"], "telecom",
                      "Billing records mediated in seconds instead of hours."),
    "tableau_sales": ("Regional sales dashboards in Tableau",
                      ["Tableau", "SQL", "Excel"], "retail",
                      "Sales managers self-serve weekly figures."),
    "informatica_erp": ("Oracle ERP data migration with Informatica",
                        ["Informatica PowerCenter", "Oracle", "ETL"], "banking",
                        "Ledger history moved with zero reconciliation breaks."),
    "faq_chatbot": ("Scripted delivery-status chatbot",
                    ["Dialogflow", "Chatbot", "Node.js"], "retail",
                    "Deflected routine tracking questions from the contact centre."),
}


def _fill(template, rng, sector_key):
    sector = SECTORS[sector_key]
    return template.format(
        a_sector=("an " if sector[0] in "aeiou" else "a ") + sector,
        sector=sector,
        pct=rng.randint(15, 60),
        n=rng.randint(3, 14),
    )


def build_projects(count, rng):
    """Specialism references first, so truncation never drops a guaranteed match."""
    slots = []
    for spec in SPECIALISMS:
        for variant in spec["projects"]:
            slots.append(("spec", spec, variant))
    for key in DISTRACTOR_PROJECTS:
        slots.append(("distractor", key, DISTRACTOR_PROJECTS[key]))
    for key in BENCH_PROJECTS:
        slots.append(("bench", key, BENCH_PROJECTS[key]))
    # Extra requested projects reuse specialism variants in a different sector.
    while len(slots) < count:
        spec = rng.choice(SPECIALISMS)
        slots.append(("spec", spec, rng.choice(spec["projects"])))
    slots = slots[:count]

    projects = []
    for kind, owner, variant in slots:
        if kind == "spec":
            summary, stack, outcome = variant
            sector_key = rng.choice(owner["sectors"])
            record = {
                "client_sector": SECTORS[sector_key],
                "summary": _fill(summary, rng, sector_key),
                "tech_stack": list(stack),
                "outcome": _fill(outcome, rng, sector_key),
            }
            tag = owner["key"]
        elif kind == "distractor":
            summary, stack, sector_key, outcome = variant
            record = {
                "client_sector": SECTORS[sector_key],
                "summary": summary,
                "tech_stack": list(stack),
                "outcome": outcome,
            }
            tag = owner
        else:
            summary, stack, outcome = variant
            sector_key = rng.choice(BENCH_SECTORS[owner])
            record = {
                "client_sector": SECTORS[sector_key],
                "summary": _fill(summary, rng, sector_key),
                "tech_stack": list(stack),
                "outcome": _fill(outcome, rng, sector_key),
            }
            tag = owner
        projects.append((tag, record))

    # IDs are assigned after shuffling so an ID never hints at its topic.
    rng.shuffle(projects)
    return [
        (tag, {"project_id": f"PRJ-{i + 1:04d}", **record})
        for i, (tag, record) in enumerate(projects)
    ]


def _past_project(record, description, tech_stack):
    # The schema has no project_id on a CV's past project, so the link to a
    # project record is textual: project_name repeats that record's summary.
    return {
        "project_name": record["summary"],
        "client_sector": record["client_sector"],
        "description": description,
        "tech_stack": tech_stack,
    }


def _pick_by_tag(projects_by_tag, tag, rng):
    candidates = projects_by_tag.get(tag, [])
    return rng.choice(candidates) if candidates else None


def build_cvs(count, projects, rng):
    projects_by_tag = {}
    for tag, record in projects:
        projects_by_tag.setdefault(tag, []).append(record)

    people = []
    for spec in SPECIALISMS:
        for slot in range(CVS_PER_SPECIALISM):
            people.append(("spec", spec, slot))
    for entry in DISTRACTOR_CVS:
        people.append(("distractor", entry, 0))
    for entry in WEB_MOBILE_BENCH:
        people.append(("bench", entry, 0))
    while len(people) < count:
        people.append(("bench", rng.choice(WEB_MOBILE_BENCH), 0))
    people = people[:count]
    rng.shuffle(people)

    cvs = []
    for index, (kind, owner, slot) in enumerate(people, start=1):
        if kind == "spec":
            title, low, high = owner["roles"][slot % len(owner["roles"])]
            years = rng.randint(low, high)
            skills = owner["skills"] + rng.sample(owner["extra"], k=rng.randint(1, 3))
            past = []
            for record in rng.sample(projects_by_tag[owner["key"]], k=rng.randint(1, 2)):
                past.append(_past_project(
                    record, owner["cv_desc"][len(past) % len(owner["cv_desc"])],
                    [t for t in record["tech_stack"] if t in skills] or record["tech_stack"][:2],
                ))
            # Roughly one specialist in three also has a web/mobile engagement,
            # which is what a consultancy CV actually looks like.
            if rng.random() < 0.35:
                bench_tag = rng.choice(list(BENCH_PROJECTS))
                record = _pick_by_tag(projects_by_tag, bench_tag, rng)
                if record:
                    past.append(_past_project(
                        record, "Supported the delivery team on integration and data tasks.",
                        record["tech_stack"][:2],
                    ))
            certs = [rng.choice(owner["certs"])] if years >= 4 or rng.random() < 0.3 else []
            role = title
        elif kind == "distractor":
            role, years, skills, description, sector_key, tag = owner
            skills = list(skills)
            record = _pick_by_tag(projects_by_tag, tag, rng) if tag else None
            if record:
                past = [_past_project(record, description, skills[:3])]
            else:
                past = [{
                    "project_name": _fill("Internal delivery for {a_sector} client", rng, sector_key),
                    "client_sector": SECTORS[sector_key],
                    "description": description,
                    "tech_stack": skills[:3],
                }]
            certs = []
        else:
            role, bench_skills, bench_tag = owner
            years = rng.randint(1, 12)
            skills = list(bench_skills)
            record = _pick_by_tag(projects_by_tag, bench_tag, rng)
            past = [_past_project(
                record,
                f"Worked as {role.lower()} on design, build and release.",
                [t for t in record["tech_stack"] if t in skills] or skills[:2],
            )]
            certs = []

        cvs.append({
            "cv_id": f"CV-{index:04d}",
            "name": f"Employee_{index:03d}",
            "role": role,
            "years_experience": years,
            "skills": skills,
            "past_projects": past,
            "certifications": certs,
        })
    return cvs


def main():
    parser = argparse.ArgumentParser(description="Generate fake CV and project data.")
    parser.add_argument("--cvs", type=int, default=59, help="Number of fake CVs to generate")
    parser.add_argument("--projects", type=int, default=35, help="Number of fake projects to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cv-out", type=str, default="fake_cvs.json", help="Output path for CVs")
    parser.add_argument("--project-out", type=str, default="fake_projects.json", help="Output path for projects")
    args = parser.parse_args()

    guaranteed_cvs = len(SPECIALISMS) * CVS_PER_SPECIALISM
    guaranteed_projects = sum(len(s["projects"]) for s in SPECIALISMS)
    if args.cvs < guaranteed_cvs or args.projects < guaranteed_projects:
        print(f"Warning: below {guaranteed_cvs} CVs / {guaranteed_projects} projects some "
              "tenders lose their guaranteed matches; check_coverage.py will fail.")

    rng = random.Random(args.seed)
    tagged_projects = build_projects(args.projects, rng)
    cvs = build_cvs(args.cvs, tagged_projects, rng)
    projects = [record for _, record in tagged_projects]

    # newline="\n": the same seed must produce identical bytes on Windows and Linux.
    with open(args.cv_out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cvs, f, indent=2, ensure_ascii=False)

    with open(args.project_out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(projects, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(cvs)} fake CVs -> {args.cv_out}")
    print(f"Generated {len(projects)} fake projects -> {args.project_out}")


if __name__ == "__main__":
    main()

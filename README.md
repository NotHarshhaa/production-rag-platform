\# Production RAG Platform



Build, evaluate, and operate production-ready Retrieval-Augmented Generation (RAG) applications with document ingestion, hybrid retrieval, grounded responses, evaluation, and observability.



\[!\[Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python\&logoColor=white)](https://www.python.org/)

\[!\[FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi\&logoColor=white)](https://fastapi.tiangolo.com/)

\[!\[Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker\&logoColor=white)](https://www.docker.com/)

\[!\[License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)



> An open-source reference implementation for building reliable, observable, and evaluable RAG systems.



\## Overview



Building a document chatbot is relatively straightforward. Building a RAG application that retrieves the right information, cites its sources, handles failures, and can be evaluated in production is a different challenge.



\*\*Production RAG Platform\*\* aims to bridge that gap by providing a modular foundation for the entire RAG lifecycle:



\- Document ingestion and processing

\- Chunking, embeddings, and vector indexing

\- Semantic and hybrid retrieval

\- Context-grounded LLM responses with citations

\- Retrieval and answer quality evaluation

\- Latency, token usage, and error monitoring

\- Containerized deployment and automated testing



The project is designed to evolve from a single-machine MVP into a scalable RAG service.



\## Key Features



\### Document ingestion



\- Upload and index PDF, TXT, and Markdown documents.

\- Extract text and split documents into configurable chunks.

\- Preserve source metadata, page numbers, and document identifiers.

\- Generate embeddings and store them in a vector database.

\- Support document re-indexing and duplicate detection.



\### Retrieval engine



\- Semantic search using vector embeddings.

\- Configurable chunk size and overlap.

\- Metadata filtering and document-level retrieval.

\- Optional hybrid retrieval combining keyword and semantic search.

\- Optional reranking to improve the relevance of retrieved passages.



\### Grounded question answering



\- Generate answers using retrieved document context.

\- Return source citations alongside answers.

\- Support configurable prompts and model providers.

\- Handle insufficient context without inventing an answer.

\- Provide an optional streaming response interface.



\### Evaluation and quality assurance



\- Evaluate retrieval relevance and context coverage.

\- Measure answer faithfulness and answer relevance.

\- Maintain a small, versioned evaluation dataset.

\- Compare retrieval configurations and prompt versions.

\- Run regression tests before deploying changes.



\### Observability



\- Track request latency, errors, and throughput.

\- Record retrieval latency and retrieved document counts.

\- Monitor token consumption and model usage where provider data is available.

\- Export application metrics to Prometheus.

\- Visualize operational metrics in Grafana.

\- Keep sensitive document content out of logs by default.



\## Architecture



```mermaid

flowchart TD

&#x20;   A\[User / Client] --> B\[FastAPI]

&#x20;   B --> C\[Query Processing]

&#x20;   C --> D\[Embedding Model]

&#x20;   D --> E\[Vector Database]

&#x20;   C --> F\[Keyword Search]

&#x20;   E --> G\[Hybrid Retrieval]

&#x20;   F --> G

&#x20;   G --> H\[Optional Reranker]

&#x20;   H --> I\[Context Builder]

&#x20;   I --> J\[LLM Provider]

&#x20;   J --> K\[Grounded Answer + Citations]

&#x20;   K --> A



&#x20;   L\[PDF / TXT / Markdown] --> M\[Document Ingestion]

&#x20;   M --> N\[Text Extraction \& Chunking]

&#x20;   N --> O\[Embedding Generation]

&#x20;   O --> E



&#x20;   B --> P\[Metrics \& Traces]

&#x20;   G --> P

&#x20;   J --> P

&#x20;   P --> Q\[Prometheus / Grafana]

&#x20;   K --> R\[Evaluation Pipeline]

```



\## Technology Stack



| Component | Recommended technology |

|---|---|

| Language | Python 3.11+ |

| API | FastAPI, Pydantic |

| Document processing | PyMuPDF, standard text parsers |

| Embeddings | Sentence Transformers or a provider API |

| Vector database | PostgreSQL with pgvector or Qdrant |

| Retrieval | Custom retrieval pipeline with optional BM25 |

| Reranking | Cross-encoder or a compatible reranking API |

| LLM | Pluggable hosted or local model |

| Evaluation | pytest and custom evaluation datasets; optional RAGAS |

| Observability | OpenTelemetry, Prometheus, Grafana |

| Persistence | PostgreSQL |

| Packaging | Docker, Docker Compose |

| CI/CD | GitHub Actions |



The initial implementation should choose one vector database and one default model configuration. Alternative providers can be added behind interfaces rather than implemented all at once.



\## Project Structure



```text

production-rag-platform/

├── app/

│   ├── main.py

│   ├── api/

│   │   ├── routes/

│   │   │   ├── documents.py

│   │   │   ├── query.py

│   │   │   └── health.py

│   │   └── dependencies.py

│   ├── core/

│   │   ├── config.py

│   │   └── logging.py

│   ├── ingestion/

│   │   ├── loaders.py

│   │   ├── chunking.py

│   │   └── pipeline.py

│   ├── retrieval/

│   │   ├── embeddings.py

│   │   ├── vector\_store.py

│   │   ├── hybrid\_search.py

│   │   └── reranker.py

│   ├── generation/

│   │   ├── prompts.py

│   │   └── answer.py

│   ├── evaluation/

│   │   ├── datasets.py

│   │   ├── metrics.py

│   │   └── runner.py

│   └── observability/

│       ├── metrics.py

│       └── tracing.py

├── tests/

│   ├── unit/

│   ├── integration/

│   └── evaluation/

├── evals/

│   └── sample\_questions.jsonl

├── scripts/

│   └── ingest\_documents.py

├── docker/

├── docs/

│   ├── architecture.md

│   └── evaluation.md

├── .github/

│   └── workflows/

│       └── ci.yml

├── .env.example

├── .gitignore

├── Dockerfile

├── compose.yaml

├── pyproject.toml

├── LICENSE

└── README.md

```



This is the target structure; modules and infrastructure files should be added as the implementation progresses.



\## Getting Started



\### Prerequisites



\- Python 3.11 or newer

\- Git

\- Docker and Docker Compose

\- Access to a compatible LLM and embedding provider, or locally runnable models



\### 1. Clone the repository



```bash

git clone https://github.com/YOUR\_USERNAME/production-rag-platform.git

cd production-rag-platform

```



\### 2. Create a virtual environment



```bash

python -m venv .venv

```



Activate it on Linux or macOS:



```bash

source .venv/bin/activate

```



On Windows PowerShell:



```powershell

.venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



Once the project dependencies have been defined in `pyproject.toml`:



```bash

pip install -e ".\[dev]"

```



\### 4. Configure environment variables



Create your local `.env` from `.env.example`.



Example configuration:



```dotenv

APP\_ENV=development

API\_HOST=0.0.0.0

API\_PORT=8000



DATABASE\_URL=postgresql://rag:rag@localhost:5432/ragdb

VECTOR\_STORE=qdrant



EMBEDDING\_PROVIDER=sentence-transformers

LLM\_PROVIDER=your\_provider

LLM\_MODEL=your\_model\_name

LLM\_API\_KEY=



LOG\_LEVEL=INFO

```



These variables are an example configuration contract. The implementation must define and validate the supported settings. Use a local secrets file and never commit real credentials.



\### 5. Start the application



After implementing the API and container configuration:



```bash

docker compose up --build

```



The development API should be available at:



\- API documentation: `http://localhost:8000/docs`

\- Health endpoint: `http://localhost:8000/health`



These endpoints are intended for the initial implementation and are not available until their routes are created.



\## Example API Workflow



The planned API includes the following operations.



| Method | Endpoint | Purpose |

|---|---|---|

| POST | `/documents/upload` | Upload a document |

| POST | `/documents/{id}/index` | Index or re-index a document |

| GET | `/documents` | List indexed documents |

| POST | `/query` | Ask a question against indexed content |

| GET | `/health` | Check service health |

| GET | `/metrics` | Expose application metrics |



Example question request:



```json

{

&#x20; "question": "How does the deployment pipeline work?",

&#x20; "top\_k": 5,

&#x20; "include\_citations": true

}

```



Example response contract:



```json

{

&#x20; "answer": "The deployment pipeline validates changes before deployment.",

&#x20; "citations": \[

&#x20;   {

&#x20;     "document\_id": "doc\_001",

&#x20;     "page": 4,

&#x20;     "chunk\_id": "chunk\_018"

&#x20;   }

&#x20; ],

&#x20; "retrieval": {

&#x20;   "chunks\_returned": 5

&#x20; }

}

```



These examples define the intended API contract, not a claim that the endpoints are already implemented.



\## Evaluation Strategy



A RAG system should be evaluated at both the retrieval and generation layers.



| Area | Example measurement |

|---|---|

| Retrieval quality | Recall@K, Precision@K, MRR |

| Context quality | Context relevance and coverage |

| Answer quality | Faithfulness and answer relevance |

| Grounding | Citation correctness and evidence coverage |

| Reliability | Request success rate and error rate |

| Performance | p50/p95 latency and throughput |

| Cost | Tokens and estimated cost per request |



Start with a small, manually reviewed question-answer dataset. Record a baseline before changing chunking, embeddings, retrieval algorithms, or prompts.



Automated evaluation scores are useful signals, not proof that every answer is correct. Validate important results against human-reviewed examples.



\## Security and Reliability



The project should account for common RAG-specific risks:



\- Validate file types and enforce upload size limits.

\- Treat retrieved documents as untrusted input.

\- Defend against prompt injection contained in documents.

\- Enforce document-level authorization before retrieval.

\- Keep API keys and credentials out of source control.

\- Avoid logging private document text or secrets.

\- Apply timeouts, retries, and rate limits to external model calls.

\- Prevent one user's documents from leaking into another user's results.

\- Use access controls and retention policies appropriate to the deployment.



RAG does not automatically prevent prompt injection or guarantee factual responses. Production deployments need explicit security controls and testing.



\## Development Roadmap



\### Phase 1 — Working RAG MVP

\- \[ ] FastAPI application and health endpoint

\- \[ ] PDF, TXT, and Markdown ingestion

\- \[ ] Chunking and embeddings

\- \[ ] Vector storage and semantic search

\- \[ ] Question-answer endpoint with citations

\- \[ ] Unit and integration tests



\### Phase 2 — Retrieval Improvements

\- \[ ] Hybrid search

\- \[ ] Metadata filtering

\- \[ ] Configurable chunking

\- \[ ] Optional reranking

\- \[ ] Document update and deletion workflows



\### Phase 3 — Evaluation

\- \[ ] Versioned evaluation dataset

\- \[ ] Retrieval metrics

\- \[ ] Answer faithfulness evaluation

\- \[ ] Citation verification

\- \[ ] Regression checks in CI



\### Phase 4 — Production Operations

\- \[ ] Docker Compose deployment

\- \[ ] Structured logs and distributed tracing

\- \[ ] Prometheus metrics and Grafana dashboards

\- \[ ] Request timeouts and rate limiting

\- \[ ] Token usage and cost tracking



\### Phase 5 — Advanced Capabilities

\- \[ ] Streaming responses

\- \[ ] Background ingestion workers

\- \[ ] Authentication and multi-user access

\- \[ ] Pluggable model and vector-store adapters

\- \[ ] Optional agentic retrieval

\- \[ ] Kubernetes deployment examples



\## Contributing



Contributions are welcome!



1\. Fork the repository.

2\. Create a feature branch.

3\. Add tests for new functionality.

4\. Run formatting, linting, and tests.

5\. Open a pull request describing the change.



Potential contribution areas include retrieval algorithms, evaluation datasets, provider integrations, observability, security, and deployment automation.



\## License



This project is intended to use the MIT License. Add the `LICENSE` file before publishing a release.



\## Author



\*\*YOUR NAME\*\*



\- GitHub: \[@YOUR\_USERNAME](https://github.com/YOUR\_USERNAME)



If you find this project useful, consider giving it a star and contributing improvements.



\---



\*\*Build RAG systems that can be evaluated, observed, and operated — not just demonstrated.\*\*




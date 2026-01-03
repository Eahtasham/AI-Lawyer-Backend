# AI Council Architecture Diagram

This diagram visualizes the flow of information through the **AI Council** system for Indian Law.

```mermaid
graph TD
    %% Styling
    classDef user fill:#f9f,stroke:#333,stroke-width:2px;
    classDef db fill:#e1f5fe,stroke:#0277bd,stroke-width:2px;
    classDef system fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef agent fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef synthesis fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    User([User Query]) ::: user --> API[FastAPI Service] ::: system
    
    subgraph "Stage 1: Ingestion & Retrieval"
        API --> embed[Embedding Model]
        embed --> Qdrant[(Qdrant Vector DB)] ::: db
        Qdrant -- "Top-K Relevant Chunks (Indian Laws)" --> Context[Context Aggregator]
    end

    Context --> CouncilManager[Council Orchestrator] ::: system

    subgraph "Stage 2: The AI Council (Parallel Agents)"
        CouncilManager -->|Context + Query| CM1[Constitutional Expert] ::: agent
        CouncilManager -->|Context + Query| CM2[Statutory Analyst] ::: agent
        CouncilManager -->|Context + Query| CM3[Case Law Researcher] ::: agent
        CouncilManager -->|Context + Query| CM4[Devil's Advocate] ::: agent
        
        CM1 -- "Perspective: Fundamental Rights" --> Aggregator[Response Aggregator]
        CM2 -- "Perspective: IPC/BNS Definitions" --> Aggregator
        CM3 -- "Perspective: Judicial Precedents" --> Aggregator
        CM4 -- "Perspective: Counter-arguments" --> Aggregator
    end

    subgraph "Stage 3: Synthesis & Judgment"
        Aggregator --> Chairman[Chairman AI (Meta-Learner)] ::: synthesis
        Chairman -- "Synthesize & Resolve Conflicts" --> FinalResult[Final Authoritative Response] ::: synthesis
    end

    FinalResult --> API
    API --> User
```

## Flow Description

1.  **User Query:** The user submits a legal question.
2.  **Retrieval:** The system embeds the query and retrieves relevant legal texts (Acts, Sections) from **Qdrant**.
3.  **The Council:** The context is sent purely to 4 specialized agents:
    *   *Constitutional Expert*
    *   *Statutory Analyst*
    *   *Case Law Researcher*
    *   *Devil's Advocate*
4.  **Synthesis:** The **Chairman AI** reviews all 4 unique perspectives, resolves contradictions (e.g., between strict statute and case law), and issues a final, cited response.

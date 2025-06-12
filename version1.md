graph TD
    A[parse_query] --> B[validate_info]
    B -->|incomplete| C[ask_missing_info]
    B -->|needs_weather| D[get_weather_data]
    B -->|complete| E[generate_response]
    C -->|interrupt| A
    D --> F[get_medical_info]
    F --> E

    subgraph "State Management"
        G[AgentState]
        G -->|messages| A
        G -->|parsed_query| B
        G -->|validation_status| B
        G -->|weather_data| D
        G -->|medical_info| F
    end

    subgraph "Tools"
        H[QueryParserTool]
        I[WeatherTool]
        J[MedicalResearchTool]
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#fbb,stroke:#333,stroke-width:2px
    style D fill:#bfb,stroke:#333,stroke-width:2px
    style E fill:#fbf,stroke:#333,stroke-width:2px
    style F fill:#bfb,stroke:#333,stroke-width:2px
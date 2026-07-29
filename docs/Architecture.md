# Architecture Documentation

The **Employee Attrition Prediction & HR Analytics Platform** employs a modern, decoupled microservices architecture. It separates the presentation layer (Streamlit) from the business and machine learning logic (FastAPI) to ensure scalability, maintainability, and ease of deployment.

## High-Level Architecture Diagram

```mermaid
flowchart TD
    Client((HR Executive))
    
    subgraph Frontend [Presentation Layer]
        UI[Streamlit Application\n(Port 8501)]
    end
    
    subgraph Backend [Application Layer]
        API[FastAPI Server\n(Port 8000)]
        Router[API Routers\nPredict, Simulate, Analytics]
        Services[Business Services\nML, Rules, DB]
    end
    
    subgraph Data [Data Layer]
        Model[(XGBoost Models)]
        DB[(SQLite Database)]
    end

    Client -->|Web Browser| UI
    UI <-->|REST HTTP Requests| API
    API --> Router
    Router --> Services
    Services <--> Model
    Services <--> DB
```

## 1. Frontend (Streamlit)
Located in `App/`, the frontend acts exclusively as a consumer of the FastAPI backend. 
- **Pages**: Divided into specialized modules (`12_Department_Analytics.py`, `13_Employee_Comparison.py`, etc.).
- **Visuals**: Uses Plotly for highly interactive visualizations (Radar charts, Bar charts).
- **Communication**: Interacts with the backend via HTTP requests using Python's `httpx` or `requests` libraries.

## 2. Backend (FastAPI)
Located in `api/`, the backend serves as the core intelligence engine.
- **`main.py`**: The entry point, configuring CORS, global logging middleware, and global exception handlers.
- **`models/schemas.py`**: Defines strict Pydantic models for incoming data. This ensures invalid data (e.g., passing a string for a numerical age) is caught before it reaches the ML model, returning a detailed `HTTP 422 Validation Error`.
- **`routes/`**: Exposes RESTful endpoints for modular interactions (`/predict`, `/simulate`, `/analytics`, `/reports`).
- **`services/`**: 
    - `ml_service.py`: Loads the serialized XGBoost model and Encoders into memory on application startup. Exposes methods for prediction and SHAP calculation.
    - `recommendation.py`: The static HR rule engine that analyzes the output of the ML model to generate contextual recommendations.
    - `db_service.py`: A DAO (Data Access Object) wrapping interactions with SQLite.

## 3. Data Layer
- **Machine Learning**: Pre-trained XGBoost classifiers and Scikit-Learn transformers stored in the `models/` directory as serialized `.pkl` files.
- **Database**: A local SQLite database (`employee_predictions.db`) stores historical predictions to power the Department Analytics and Workforce Planning dashboards.

## Docker & Orchestration
Both the frontend and backend are containerized separately:
- **`backend.Dockerfile`**: Uses Uvicorn to serve the FastAPI app.
- **`frontend.Dockerfile`**: Uses Streamlit to serve the UI.
- **`docker-compose.yml`**: Mounts shared volumes ensuring both containers can read/write to the SQLite database and access the `models/` directory synchronously.

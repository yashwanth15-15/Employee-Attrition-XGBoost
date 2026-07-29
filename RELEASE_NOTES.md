# Release Notes v2.0.0

## Highlights
Welcome to version 2.0.0 of the Employee Attrition Prediction Platform! This major release completely overhauls the backend architecture, introducing robust JWT Authentication, Role-Based Access Control, and a highly scalable SQLAlchemy ORM mapping supporting both SQLite and PostgreSQL.

## Breaking Changes
- Previous raw SQLite database files are incompatible with the new SQLAlchemy schema layout.
- The `pages/` directory has been restructured to `views/` to enforce unified Streamlit navigation routing.

## New Features
- Full JWT Authentication workflow.
- Export to PDF Reporting capabilities.
- Advanced Department Analytics visualizations.
- Batch processing predictions from Streamlit.

## Bug Fixes
- Addressed issues with multi-select deletions in the Prediction History UI.
- Rectified health check endpoint false negatives.
- Streamlit Session State logic securely patched.

## Performance & Security
- Integrated `bcrypt` for secure password hashing.
- Optimized REST API structure using FastAPI dependency injections.

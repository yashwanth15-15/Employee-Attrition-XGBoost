# Deployment Guide v2.0.0

The application leverages a flexible SQLAlchemy ORM allowing local testing with SQLite and production deployment with PostgreSQL.

## Docker
The platform provides a `Dockerfile` and `docker-compose.yml` for unified deployment of the FastAPI backend and Streamlit frontend.

## Render Deployment
1. Connect GitHub repository to Render.
2. Set Environment Variables:
   - `DATABASE_URL`: Your Managed PostgreSQL URI
   - `JWT_SECRET_KEY`: A secure random string
3. Deploy!

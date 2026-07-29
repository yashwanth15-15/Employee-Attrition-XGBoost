# Deployment Guide

This document outlines the steps required to deploy the Employee Attrition Prediction Platform to modern cloud hosting providers. Because the application is decoupled into a FastAPI backend and a Streamlit frontend, they can be deployed independently for maximum scalability.

## 1. Deploying the Frontend (Streamlit Community Cloud)

Streamlit Community Cloud is the optimal platform for hosting the Streamlit frontend. It is free and natively supports GitHub repositories.

**Pre-requisites:**
Before deploying, ensure your `app/app.py` script makes API calls to your hosted FastAPI backend URL, rather than `localhost:8000`. You can handle this via environment variables.

**Steps:**
1. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **New app**.
3. Select this GitHub repository and the `main` branch.
4. Set the **Main file path** to `app/app.py`.
5. Click **Advanced settings** and define your Environment Variables:
   ```env
   BACKEND_API_URL=https://your-fastapi-backend-url.onrender.com
   ```
6. Click **Deploy**. Your dashboard will be live in minutes.

---

## 2. Deploying the Backend (Render)

Render offers a generous free tier and native Docker support, making it perfect for the FastAPI backend.

**Steps:**
1. Log in to [Render](https://render.com/).
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository.
4. Render will automatically detect the repository. In the settings, change the **Runtime** to `Docker`.
5. Specify the Dockerfile path as `backend.Dockerfile`.
6. Set the **Start Command** (optional, handled by Dockerfile, but can be forced):
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port $PORT
   ```
7. Click **Create Web Service**. 
8. Once deployed, note the Render URL and update your Streamlit Environment Variables.

---

## 3. Deploying the Full Stack (Railway)

Railway allows you to deploy the entire `docker-compose.yml` stack simultaneously in a single project.

**Steps:**
1. Log in to [Railway](https://railway.app/).
2. Click **New Project** -> **Deploy from GitHub repo**.
3. Select your repository.
4. Railway will automatically parse the `docker-compose.yml` file and provision two separate services: `backend` and `frontend`.
5. Once built, go to the settings for both services and **Generate Domain**.
6. In the `frontend` service settings, add an environment variable pointing to the newly generated `backend` domain:
   ```env
   BACKEND_API_URL=https://backend-production-xyz.up.railway.app
   ```
7. Redeploy the frontend to apply the environment variable.

---

## Important Considerations for Production
*   **Database Persistence**: In ephemeral cloud platforms (like Render Free Tier or Heroku), the local `employee_predictions.db` SQLite file will be wiped every time the server restarts. For production, migrate the DB connection in `app/database.py` and `api/config.py` to a managed PostgreSQL database (e.g., Supabase or Neon).
*   **CORS**: Ensure that the `api/main.py` CORS configuration includes the production URL of your frontend (e.g., `https://your-app.streamlit.app`).

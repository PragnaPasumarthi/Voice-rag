@echo off
echo Starting Streamlit UI...
echo Make sure the API server is running on port 8000
echo.
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0

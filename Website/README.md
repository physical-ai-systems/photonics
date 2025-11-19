# Task 1 - FastAPI Website

## How to Run

1. Install dependencies:
   ```bash
   pip install fastapi uvicorn
   ```

2. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

3. The API will be available at `http://127.0.0.1:8000`

4. Open `template/index.html` in your browser to view the frontend

## API Endpoints

- `GET /` - Check if the backend is running
- `POST /calculate` - Perform simulation calculations

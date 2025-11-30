
# FastAPI Project

This project is built with [FastAPI](https://fastapi.tiangolo.com/) and uses [Uvicorn](https://www.uvicorn.org/) as the ASGI server.

## Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/Eahtasham/AI-Lawyer-Backend.git
   cd AI-Lawyer-Backend
   ````

2. **Create and activate virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Linux / macOS
   venv\Scripts\activate       # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the development server**

   ```bash
   uvicorn app.main:app --reload
   ```

   * `main` refers to the `main.py` file (update if different).
   * `app` is the FastAPI instance.

5. **Access the application**

   * API root: [http://127.0.0.1:8000](http://127.0.0.1:8000)
   * Interactive docs (Swagger UI - Recommanded): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   * Alternative docs (ReDoc): [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## Notes

* Always activate the virtual environment before running commands.
* Add any new dependencies using:

  ```bash
  pip install <package>
  pip freeze > requirements.txt
  ```
* To stop the server, press `CTRL + C`.


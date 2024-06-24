from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from hex import generate_k_ring

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate_random_k_ring")
async def generate_random_k_ring(request: Request):
    hexs = generate_k_ring(await request.json())
    return JSONResponse(content={"hexagons": hexs})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0")

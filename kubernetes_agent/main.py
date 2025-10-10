# __main__.py
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from k8s_agent_executor import k8sAgentExecutor
from agent_card import public_agent_card



# Create the main ASGI app
request_handler = DefaultRequestHandler(
    agent_executor=k8sAgentExecutor(),
    task_store=InMemoryTaskStore(),
)

application = A2AStarletteApplication(
    agent_card=public_agent_card,
    http_handler=request_handler,
)

# Optionally mount a FastAPI app for /.well-known/agent-card.json
from fastapi import FastAPI
from fastapi.responses import JSONResponse

agent_card_app = FastAPI()

@agent_card_app.get("/agent-card.json")
def agent_card():
    return JSONResponse(public_agent_card.model_dump())

app = application.build()
app.mount("/.well-known", agent_card_app)

# Expose as 'server' for uvicorn
server = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(server, host="0.0.0.0", port=8080)

import asyncio
import json
import uuid
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .discovery import DiscoveryAgent

app = FastAPI()

# Global limits
import datetime
job_semaphore = asyncio.Semaphore(1)
daily_limit = {"date": None, "count": 0}
import os
MAX_DAILY_JOBS = int(os.environ.get("MAX_DAILY_JOBS", 5))

# In-memory registry for jobs and their queues
jobs: Dict[str, asyncio.Queue] = {}
background_tasks = set()

class DiscoverRequest(BaseModel):
    datasets: list[str] = ["Cora", "ETTh1"]
    access_code: str = ""

@app.post("/api/discover")
async def start_discovery(req: DiscoverRequest):
    from .tasks import TASKS
    for ds in req.datasets:
        if ds not in TASKS:
            raise HTTPException(status_code=400, detail=f"Dataset {ds} is not supported.")
            
    import os
    import hmac
    
    expected_code = os.environ.get("ACCESS_CODE")
    if not expected_code:
        raise HTTPException(status_code=500, detail="Server misconfigured: ACCESS_CODE not set")
        
    if not req.access_code or not hmac.compare_digest(req.access_code.encode(), expected_code.encode()):
        raise HTTPException(status_code=403, detail="Invalid access code")
        
    today = datetime.date.today()
    if daily_limit["date"] != today:
        daily_limit["date"] = today
        daily_limit["count"] = 0
        
    if daily_limit["count"] >= MAX_DAILY_JOBS:
        raise HTTPException(status_code=429, detail="Daily job limit exceeded")
        
    if job_semaphore.locked():
        raise HTTPException(status_code=429, detail="A job is already running. Try again later.")
        
    daily_limit["count"] += 1
    
    job_id = str(uuid.uuid4())
    queue = asyncio.Queue()
    jobs[job_id] = queue
    
    # Run the discovery loop in a background thread or task.
    # We use asyncio.to_thread because the DiscoveryAgent uses synchronous httpx/Tavily calls.
    def discovery_worker():
        agent = DiscoveryAgent()
        def on_event(evt):
            # Put the event into the queue in a thread-safe manner
            if job_id not in jobs:
                raise RuntimeError("Client disconnected, aborting discovery loop.")
            try:
                loop.call_soon_threadsafe(queue.put_nowait, evt)
            except Exception:
                pass
                
        try:
            agent.discover(req.datasets, on_event=on_event)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "done"})
            loop.call_soon_threadsafe(job_semaphore.release)
            
    loop = asyncio.get_running_loop()
    await job_semaphore.acquire()
    # Start the worker thread
    task = asyncio.create_task(asyncio.to_thread(discovery_worker))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    
    return {"job_id": job_id}


@app.get("/api/discover/{job_id}/stream")
async def stream_discovery(job_id: str, request: Request):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    queue = jobs[job_id]
    
    async def event_generator():
        try:
            while True:
                # Disconnect if client leaves
                if await request.is_disconnected():
                    break
                    
                try:
                    # Wait for an event with a 15-second timeout for keepalive
                    evt = await asyncio.wait_for(queue.get(), timeout=15.0)
                    
                    if evt.get("type") == "done":
                        break
                        
                    yield f"data: {json.dumps(evt)}\n\n"
                    
                except asyncio.TimeoutError:
                    # Emit keepalive to prevent proxy disconnects
                    yield ": keepalive\n\n"
        finally:
            # Cleanup when the stream is closed
            if job_id in jobs:
                del jobs[job_id]
                
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

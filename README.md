# Neurovision-AI-MRI
Neurovision-AI-MRI is an explainable AI platform that transforms neural network activity into a real-time anatomical brain simulation.

Instead of treating AI as a black box, this system exposes its internal cognitive process by combining:

• YOLOv10 object detection  
• CLIP semantic understanding  
• Grad-CAM visual explainability  
• PyTorch neural activation extraction  
• MRI-based anatomical brain mapping  
• Interactive 3D brain visualization  

The result is a digital brain that reveals how AI perceives, interprets, and decides — mapped directly onto real cortical regions.

## Production deployment notes (Render + Gunicorn)

For CLIP CPU inference on Render Standard plans, the most stable setup is usually:

- `workers=1` (avoid multiple model copies in RAM)
- `preload_app=true` (load model once in master before fork)
- `CLIP_PRELOAD_ON_START=1` (warm model at startup, not at first request)
- longer `timeout` (e.g. 180s) to absorb cold boot and heavy images

Suggested start command:

```bash
gunicorn app:app -c gunicorn.conf.py
```

Useful env vars:

- `GUNICORN_PRELOAD=1`
- `WEB_CONCURRENCY=1`
- `GUNICORN_THREADS=2`
- `GUNICORN_TIMEOUT=180`
- `CLIP_PRELOAD_ON_START=1`

Readiness endpoint is available at `/ready` and returns `503` until model is loaded.

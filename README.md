# VisionGuide AI (Cloud Prototype)

## Files
- `streamlit_app.py`: Streamlit app
- `yolo26s.pt`: YOLO26 Small model weights
- `requirements.txt`: Python dependencies

## Run locally
```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app receives camera frames from the visitor's browser using WebRTC.
Camera images are transmitted to the server for inference. Users must grant
camera permission and should not use the app with sensitive/private scenes.

## Deployment notes
- Add `streamlit_app.py`, `requirements.txt`, and `yolo26s.pt` to the deployment repository.
- Select `streamlit_app.py` as the app entry point.
- WebRTC hosting may require a STUN/TURN configuration depending on the hosting
  platform and network. Test from a phone on mobile data as well as Wi-Fi.
- YOLO inference runs on CPU in this prototype. A small cloud instance may be
  slow or run out of memory. Measure performance before presenting it as live
  real-time service.

## Safety
This is an experimental object-detection demonstration, not a mobility aid.
It does not measure distance or determine whether a route is safe.

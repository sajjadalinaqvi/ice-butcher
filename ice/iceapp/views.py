# views.py
import os
import base64
import io
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from huggingface_hub import InferenceClient
from PIL import Image

# Load token
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Missing Hugging Face API key. Set HF_TOKEN in your .env or environment.")

# Initialize client (fal-ai provider is required for this model)
client = InferenceClient(
    provider="fal-ai",
    api_key=HF_TOKEN,
)

MODEL_ID = "ProomptEngineer/pe-ice-sculpture-style"


def index(request):
    return render(request, "iceapp/index.html")


@csrf_exempt
def generate_image(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    prompt = request.POST.get("prompt") or request.GET.get("prompt")
    if not prompt:
        return JsonResponse({"error": "prompt required"}, status=400)

    try:
        # Call Hugging Face Inference API
        image: Image.Image = client.text_to_image(
            prompt,
            model=MODEL_ID,
        )

        # Convert to base64 for frontend
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return JsonResponse({"image_base64": img_b64})

    except Exception as e:
        return JsonResponse({"error": f"HF request failed: {str(e)}"}, status=502)

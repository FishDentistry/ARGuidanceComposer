"""Persistent Grounding DINO subprocess with isolated Transformers dependencies."""
import base64
import io
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '.guidance-deps'))

def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    device = sys.argv[1]
    model_id = sys.argv[2]
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id, local_files_only=True).eval().to(device)
    print(json.dumps({'ready': True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            image = Image.open(io.BytesIO(base64.b64decode(request['image']))).convert('RGB')
            text = request['description'].strip().lower().rstrip('.') + '.'
            inputs = processor(images=image, text=text, return_tensors='pt').to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=request['threshold'],
                text_threshold=0.25, target_sizes=[image.size[::-1]])[0]
            candidates = [{'box': b.tolist(), 'score': float(s), 'label': label}
                          for b, s, label in zip(result['boxes'], result['scores'], result['labels'])]
            print(json.dumps({'candidates': candidates}), flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({'error': str(exc)}), flush=True)

if __name__ == '__main__':
    main()

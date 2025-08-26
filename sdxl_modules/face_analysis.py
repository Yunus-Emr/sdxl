import sys
sys.path.append("./sdxl_modules")
sys.path.append("./")
import sys, cv2, numpy as np
from diffusers.utils import load_image
from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import draw_kps

def setup_face_app(root_path="./", det_size=(640,640)):
    app = FaceAnalysis(
        name='antelopev2',
        root=root_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=det_size)
    return app

def get_face_info(app, image_path):
    face_image = load_image(image_path)
    face_info_list = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info_list, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    face_emb = face_info['embedding']
    face_kps = draw_kps(face_image, face_info['kps'])
    return face_image, face_emb, face_kps
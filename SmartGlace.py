import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time


class SmartGlaceApp:
    """Application SmartGlace - Miroir interactif avec effets realistes."""

    def __init__(self):
        # MediaPipe avec parametres optimaux
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Effets professionnels (Icones texte)
        self.effects = [
            {"name": "Normal", "icon": "Nrm", "description": "Vue originale"},
            {"name": "Glow Skin", "icon": "Glw", "description": "Peau lumineuse et lisse"},
            {"name": "Anime Eyes", "icon": "Anm", "description": "Grands yeux style anime"},
            {"name": "Color Grading", "icon": "Clr", "description": "Correction colorimetrique cinema"},
            {"name": "Portrait Mode", "icon": "Prt", "description": "Flou d'arriere-plan (Bokeh)"},
            {"name": "Beauty Filter", "icon": "Bty", "description": "Lissage peau + maquillage"},
            {"name": "Cyberpunk", "icon": "Cyb", "description": "Style futuriste neon"},
            {"name": "Vintage Film", "icon": "Vnt", "description": "Effet pellicule ancienne"},
            {"name": "3D Depth", "icon": "3D", "description": "Effet de profondeur 3D"},
            {"name": "Rainbow Prism", "icon": "Rbw", "description": "Dispersion prismatique"},
            {"name": "Crystal Face", "icon": "Crs", "description": "Facettes cristallines"},
            {"name": "Liquid Metal", "icon": "Liq", "description": "Texture metal liquide"},
            {"name": "Cartoon", "icon": "Crt", "description": "Style dessin anime"},
            {"name": "Glitch", "icon": "Glt", "description": "Effet de distorsion TV"},
            {"name": "Bulge", "icon": "Blg", "description": "Deformation (nez large)"},
            {"name": "Overlay", "icon": "Ovl", "description": "Surimpression florale"},
            {"name": "Teeth Whiten", "icon": "Tth", "description": "Souris (bouche ouverte) pour blanchir"},
            {"name": "Sleepy", "icon": "Slp", "description": "Ferme les yeux pour flouter"}
        ]
        self.current_effect = 0

      
        # Regions faciales detaillees
        self.face_regions = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'left_iris': [474, 475, 476, 477],
            'right_iris': [469, 470, 471, 472],
            'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'right_eyebrow': [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
            'nose': [1, 2, 98, 327, 4, 5, 6, 195, 197],
            'lips_upper': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
            'lips_lower': [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                         397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                         172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            'left_cheek': [116, 117, 118, 119, 123],
            'right_cheek': [345, 346, 347, 348, 352],
            'forehead': [10, 338, 297, 332, 284, 251, 389, 356, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300],
            'inner_mouth': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        }

        # Etats et historiques
        self.smoothing_buffer = deque(maxlen=5)
        self.fps_history = deque(maxlen=30)
        self.face_detected = False
        self.detection_confidence = 0.0
        self.last_landmarks = None
        
        # Textures et masques precharges
        self.beauty_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # --- NOUVEAUTES POUR EXPRESSIONS ET OVERLAY ---
        
        # Etats d'expression
        self.smile_detected = False
        self.eyes_closed = False
        self.mouth_open = False
        self.smile_threshold = 0.28
        self.ear_threshold = 0.22
        self.mouth_open_threshold = 0.05
        
        # Indices des points pour le calcul EAR (Eye Aspect Ratio)
        self.EYE_LEFT_INDICES = [33, 160, 158, 133, 153, 144]
        self.EYE_RIGHT_INDICES = [362, 385, 387, 263, 373, 380]
        
        # Texture generee pour l'overlay
        self.overlay_texture = self.generate_overlay_texture(720, 1280)
        
        # --- NOUVEAU : Compteur pour Glitch ---
        self.glitch_counter = 0

        
    # =========================================================================
    # NOUVEAUX UTILITAIRES (Expressions et Textures)
    # =========================================================================

    def generate_overlay_texture(self, h, w):
        """Genere une texture abstraite pour l'effet d'overlay."""
        base = np.zeros((h, w, 3), dtype=np.uint8)
        base = cv2.randn(base, (128), (50))
        base = cv2.applyColorMap(base, cv2.COLORMAP_TWILIGHT_SHIFTED)
        base = cv2.GaussianBlur(base, (199, 199), 0)
        return cv2.addWeighted(base, 0.7, np.zeros_like(base), 0.3, 0)

    def get_ear(self, eye_indices, landmarks, w, h):
        """Calcule le Eye Aspect Ratio (EAR) pour un oeil."""
        try:
            p1 = np.array([landmarks[eye_indices[0]].x * w, landmarks[eye_indices[0]].y * h])
            p2 = np.array([landmarks[eye_indices[1]].x * w, landmarks[eye_indices[1]].y * h])
            p3 = np.array([landmarks[eye_indices[2]].x * w, landmarks[eye_indices[2]].y * h])
            p4 = np.array([landmarks[eye_indices[3]].x * w, landmarks[eye_indices[3]].y * h])
            p5 = np.array([landmarks[eye_indices[4]].x * w, landmarks[eye_indices[4]].y * h])
            p6 = np.array([landmarks[eye_indices[5]].x * w, landmarks[eye_indices[5]].y * h])
            
            A = np.linalg.norm(p2 - p6)
            B = np.linalg.norm(p3 - p5)
            C = np.linalg.norm(p1 - p4)
            
            if C == 0: return 0.3
            ear = (A + B) / (2.0 * C)
            return ear
        except Exception:
            return 0.3

    def calculate_expression_metrics(self, landmarks, w, h):
        """Met a jour self.smile_detected, self.mouth_open et self.eyes_closed."""
        if not landmarks:
            self.smile_detected = False
            self.eyes_closed = False
            self.mouth_open = False
            return

        try:
            # Points des yeux pour normalisation
            eye_left = np.array([landmarks[self.EYE_LEFT_INDICES[0]].x * w, landmarks[self.EYE_LEFT_INDICES[0]].y * h])
            eye_right = np.array([landmarks[self.EYE_RIGHT_INDICES[0]].x * w, landmarks[self.EYE_RIGHT_INDICES[0]].y * h])
            interocular_dist = np.linalg.norm(eye_left - eye_right)
            
            if interocular_dist == 0:
                self.smile_detected = False
                self.eyes_closed = False
                self.mouth_open = False
                return

            # 1. Detection de sourire (largeur)
            lip_left = np.array([landmarks[61].x * w, landmarks[61].y * h])
            lip_right = np.array([landmarks[291].x * w, landmarks[291].y * h])
            mouth_width = np.linalg.norm(lip_left - lip_right)
            smile_ratio = mouth_width / interocular_dist
            self.smile_detected = smile_ratio > self.smile_threshold

            # 2. Detection bouche ouverte (hauteur)
            lip_top_inner = np.array([landmarks[13].x * w, landmarks[13].y * h])
            lip_bottom_inner = np.array([landmarks[14].x * w, landmarks[14].y * h])
            mouth_open_dist = np.linalg.norm(lip_top_inner - lip_bottom_inner)
            mouth_open_ratio = mouth_open_dist / interocular_dist
            self.mouth_open = mouth_open_ratio > self.mouth_open_threshold

            # 3. Detection yeux fermes (EAR)
            left_ear = self.get_ear(self.EYE_LEFT_INDICES, landmarks, w, h)
            right_ear = self.get_ear(self.EYE_RIGHT_INDICES, landmarks, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            self.eyes_closed = avg_ear < self.ear_threshold
            
        except Exception:
            self.smile_detected = False
            self.eyes_closed = False
            self.mouth_open = False


    # =========================================================================
    # UTILITAIRES PROFESSIONNELS (Existants)
    # =========================================================================
    
    def smooth_landmarks(self, landmarks):
        """Lissage temporel des landmarks pour stabilite."""
        if landmarks is None:
            return self.last_landmarks
            
        if self.last_landmarks is None:
            self.last_landmarks = landmarks
            return landmarks
         
        alpha = 0.7
        smoothed = []
        for i, lm in enumerate(landmarks):
            old_lm = self.last_landmarks[i]
            smoothed.append(type(lm)(
                x=alpha * lm.x + (1 - alpha) * old_lm.x,
                y=alpha * lm.y + (1 - alpha) * old_lm.y,
                z=alpha * lm.z + (1 - alpha) * old_lm.z
            ))
        
        self.last_landmarks = smoothed
        return smoothed
    
    def create_face_mask(self, image, landmarks):
        """Cree un masque precis du visage avec feathering."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        face_points = np.array([
            [int(landmarks[i].x * w), int(landmarks[i].y * h)]
            for i in self.face_regions['face_oval']
        ])
        
        cv2.fillPoly(mask, [face_points], 255)
        mask = cv2.GaussianBlur(mask, (21, 21), 11)
        
        return mask
    
    def apply_bilateral_beauty(self, image, mask, strength=0.8):
        """Application professionnelle du lissage de peau."""
        smooth = cv2.bilateralFilter(image, 9, 75, 75)
        smooth = cv2.ximgproc.guidedFilter(image, smooth, 8, 0.1**2)
        
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = (smooth * mask_3ch * strength + image * (1 - mask_3ch * strength)).astype(np.uint8)
        
        return result
    
    def enhance_details(self, image, strength=1.5):
        """Amelioration des details (yeux, levres)."""
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp = cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)
        
        return unsharp
    
    # =========================================================================
    # EFFETS REALISTES PROFESSIONNELS (Existants et Modifies)
    # =========================================================================
    
    def effect_glow_skin(self, image, landmarks, mask):
        """Peau lumineuse et lisse (effet Instagram)."""
        result = image.copy()
        
        result = self.apply_bilateral_beauty(result, mask, strength=0.9)
        
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        mask_f = (mask / 255.0).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + 0.2 * mask_f), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        glow = cv2.GaussianBlur(result, (0, 0), 10)
        result = cv2.addWeighted(result, 0.8, glow, 0.2, 0)
        
        detail_mask = np.zeros_like(mask)
        h, w = image.shape[:2]
        for region in ['left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow', 'lips_upper', 'lips_lower']:
            points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                                  for i in self.face_regions[region]])
            cv2.fillPoly(detail_mask, [points], 255)
        
        details = self.enhance_details(image, strength=1.8)
        detail_mask_3ch = cv2.cvtColor(detail_mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = (details * detail_mask_3ch + result * (1 - detail_mask_3ch)).astype(np.uint8)
        
        return result
    
    def effect_anime_eyes(self, image, landmarks):
        """Grands yeux style anime avec reflets realistes."""
        result = image.copy()
        h, w = result.shape[:2]
        
        for eye_region, iris_region in [('left_eye', 'left_iris'), ('right_eye', 'right_iris')]:
            eye_points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                                  for i in self.face_regions[eye_region][:8]])
            
            eye_center = eye_points.mean(axis=0).astype(int)
            eye_width = int(np.linalg.norm(eye_points[0] - eye_points[4]))
            eye_height = int(np.linalg.norm(eye_points[2] - eye_points[6]))
            
            margin = int(eye_width * 0.5)
            x1, y1 = max(0, eye_center[0] - eye_width - margin), max(0, eye_center[1] - eye_height - margin)
            x2, y2 = min(w, eye_center[0] + eye_width + margin), min(h, eye_center[1] + eye_height + margin)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            eye_roi = image[y1:y2, x1:x2].copy()
            
            scale = 1.4
            enlarged = cv2.resize(eye_roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab[:, :, 0] = np.clip(lab[:, :, 0] * 1.1, 0, 255)
            enlarged = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            
            eh, ew = enlarged.shape[:2]
            blend_mask = np.zeros((eh, ew), dtype=np.float32)
            cv2.ellipse(blend_mask, (ew // 2, eh // 2), (ew // 3, eh // 3), 0, 0, 360, 1, -1)
            blend_mask = cv2.GaussianBlur(blend_mask, (31, 31), 15)
            
            paste_x = max(0, eye_center[0] - ew // 2)
            paste_y = max(0, eye_center[1] - eh // 2)
            paste_x2 = min(w, paste_x + ew)
            paste_y2 = min(h, paste_y + eh)
            
            crop_w = paste_x2 - paste_x
            crop_h = paste_y2 - paste_y
            
            if crop_w > 0 and crop_h > 0:
                blend_mask_crop = blend_mask[:crop_h, :crop_w]
                blend_mask_3ch = np.stack([blend_mask_crop] * 3, axis=2)
                
                roi = result[paste_y:paste_y2, paste_x:paste_x2]
                blended = (enlarged[:crop_h, :crop_w] * blend_mask_3ch + 
                          roi * (1 - blend_mask_3ch)).astype(np.uint8)
                result[paste_y:paste_y2, paste_x:paste_x2] = blended
            
            if len(self.face_regions[iris_region]) >= 4:
                iris_center = np.array([
                    int(landmarks[self.face_regions[iris_region][0]].x * w),
                    int(landmarks[self.face_regions[iris_region][0]].y * h)
                ])
                
                highlight_radius = int(eye_width * 0.15)
                highlight_pos = (iris_center[0] - highlight_radius // 2, 
                               iris_center[1] - highlight_radius // 2)
                
                overlay = result.copy()
                cv2.circle(overlay, highlight_pos, highlight_radius, (255, 255, 255), -1)
                cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
                
                highlight_pos2 = (iris_center[0] + highlight_radius, 
                                iris_center[1] + highlight_radius)
                cv2.circle(overlay, highlight_pos2, highlight_radius // 2, (255, 255, 255), -1)
                cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
        
        return result
    
    def effect_color_grading(self, image):
        """Color grading cinematographique professionnel."""
        result = image.copy().astype(np.float32) / 255.0
        
        result = np.clip(result * 1.2 - 0.1, 0, 1)
        result = np.power(result, 0.9)
        
        hsv = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        
        shadow_mask = (hsv[:, :, 2] < 100).astype(np.float32)
        hsv[:, :, 0] = hsv[:, :, 0] + shadow_mask * 10
        
        highlight_mask = (hsv[:, :, 2] > 150).astype(np.float32)
        hsv[:, :, 0] = hsv[:, :, 0] - highlight_mask * 5
        
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        h, w = result.shape[:2]
        vignette = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(vignette, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 1, -1)
        vignette = cv2.GaussianBlur(vignette, (0, 0), w / 3)
        vignette = np.stack([vignette] * 3, axis=2)
        
        result = (result * (0.6 + 0.4 * vignette)).astype(np.uint8)
        
        return result
    
    def effect_portrait_mode(self, image, mask):
        """Flou d'arriere-plan style Bokeh, plus doux."""
        
        background = cv2.GaussianBlur(image, (0, 0), 15)

        gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        _ , highlight_mask = cv2.threshold(gray_bg, 200, 255, cv2.THRESH_BINARY)
        
        highlight_glow = cv2.dilate(highlight_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        highlight_glow = cv2.GaussianBlur(highlight_glow, (31, 31), 0)
        
        highlight_glow_3ch = cv2.cvtColor(highlight_glow, cv2.COLOR_GRAY2BGR)
        
        background = cv2.addWeighted(background, 1.0, highlight_glow_3ch, 0.5, 0)
        
        inv_mask = 255 - mask
        inv_mask = cv2.GaussianBlur(inv_mask, (31, 31), 15)
        inv_mask_3ch = np.stack([inv_mask] * 3, axis=2) / 255.0
        
        result = (background * inv_mask_3ch + image * (1 - inv_mask_3ch)).astype(np.uint8)
        
        face_enhanced = self.enhance_details(image, strength=1.2)
        mask_3ch = np.stack([mask] * 3, axis=2) / 255.0
        result = (face_enhanced * mask_3ch + result * (1 - mask_3ch)).astype(np.uint8)
        
        return result
    
    def effect_beauty_filter(self, image, landmarks, mask):
        """Filtre beaute complet (peau + maquillage)."""
        result = self.effect_glow_skin(image, landmarks, mask)
        h, w = result.shape[:2]
        
        overlay = result.copy()
        for cheek_region in ['left_cheek', 'right_cheek']:
            cheek_points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                                    for i in self.face_regions[cheek_region]])
            cheek_center = cheek_points.mean(axis=0).astype(int)
            cv2.circle(overlay, tuple(cheek_center), 30, (180, 100, 150), -1)
        
        overlay = cv2.GaussianBlur(overlay, (51, 51), 0)
        result = cv2.addWeighted(result, 0.8, overlay, 0.2, 0)
        
        lips_mask = np.zeros((h, w), dtype=np.uint8)
        for lip_region in ['lips_upper', 'lips_lower']:
            lip_points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                                  for i in self.face_regions[lip_region]])
            cv2.fillPoly(lips_mask, [lip_points], 255)
        
        lips_mask = cv2.GaussianBlur(lips_mask, (7, 7), 0)
        lips_overlay = result.copy()
        
        hsv = cv2.cvtColor(lips_overlay, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
        lips_overlay = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        lips_mask_3ch = np.stack([lips_mask] * 3, axis=2) / 255.0
        result = (lips_overlay * lips_mask_3ch + result * (1 - lips_mask_3ch)).astype(np.uint8)
        
        return result
    
    def effect_cyberpunk(self, image, landmarks):
        """Style cyberpunk avec neons realistes."""
        result = image.copy()
        h, w = result.shape[:2]
        
        result = cv2.convertScaleAbs(result, alpha=0.7, beta=-20)
        
        overlay = np.zeros_like(result)
        
        colors = [(255, 0, 200), (0, 255, 255), (255, 255, 0)]
        regions = ['face_oval', 'left_eye', 'right_eye', 'lips_upper', 'lips_lower']
        
        for i, region in enumerate(regions):
            points = np.array([[int(landmarks[idx].x * w), int(landmarks[idx].y * h)] 
                                  for idx in self.face_regions[region]])
            color = colors[i % len(colors)]
            cv2.polylines(overlay, [points], region == 'face_oval', color, 2, cv2.LINE_AA)
        
        glow = cv2.GaussianBlur(overlay, (15, 15), 0)
        overlay = cv2.addWeighted(overlay, 0.7, glow, 0.5, 0)
        
        result = cv2.addWeighted(result, 1.0, overlay, 0.8, 0)
        
        noise = np.random.randint(-10, 10, result.shape, dtype=np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return result
    
    def effect_vintage_film(self, image):
        """Effet pellicule vintage realiste."""
        result = image.copy()
        
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.addWeighted(result, 0.6, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.4, 0)
        
        sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])
        result = cv2.transform(result, sepia_kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        h, w = result.shape[:2]
        vignette = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(vignette, (w // 2, h // 2), (int(w * 0.4), int(h * 0.4)), 0, 0, 360, 1, -1)
        vignette = cv2.GaussianBlur(vignette, (0, 0), w / 2)
        vignette = np.stack([vignette] * 3, axis=2)
        
        result = (result * (0.3 + 0.7 * vignette)).astype(np.uint8)
        
        noise = np.random.normal(0, 15, result.shape).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        for _ in range(5):
            x = np.random.randint(0, w)
            cv2.line(result, (x, 0), (x, h), (200, 200, 200), 1)
        
        return result
    
    def effect_3d_depth(self, image, landmarks):
        """Effet de profondeur 3D avec decalage chromatique."""
        h, w = image.shape[:2]
        result = np.zeros_like(image)
        
        face_points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                               for i in self.face_regions['face_oval']])
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [face_points], 255)
        
        offset = 5
        result[:, :, 2] = np.roll(image[:, :, 2], -offset, axis=1)
        result[:, :, 1] = image[:, :, 1]
        result[:, :, 0] = np.roll(image[:, :, 0], offset, axis=1)
        result = cv2.addWeighted(result, 0.7, image, 0.3, 0)
        
        return result
    
    def effect_rainbow_prism(self, image, landmarks):
        """Dispersion prismatique arc-en-ciel."""
        h, w = image.shape[:2]
        result = image.copy()
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)
        
        rainbow = np.zeros_like(image)
        
        for i, color in enumerate([(255, 0, 0), (255, 127, 0), (255, 255, 0), 
                                   (0, 255, 0), (0, 0, 255), (75, 0, 130), (148, 0, 211)]):
            offset = (i - 3) * 2
            shifted_edges = np.roll(edges, offset, axis=1)
            rainbow[shifted_edges > 0] = color
        
        rainbow = cv2.GaussianBlur(rainbow, (15, 15), 0)
        result = cv2.addWeighted(result, 0.7, rainbow, 0.3, 0)
        
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result
    
    def effect_crystal_face(self, image, landmarks):
        """Facettes cristallines realistes."""
        h, w = image.shape[:2]
        result = image.copy()
        
        face_points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                               for i in self.face_regions['face_oval']])
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [face_points], 255)
        
        num_facets = 150
        points = []
        min_x, min_y = face_points.min(axis=0)
        max_x, max_y = face_points.max(axis=0)
        
        if min_x >= max_x or min_y >= max_y:
            return image

        for _ in range(num_facets):
            x = np.random.randint(min_x, max_x)
            y = np.random.randint(min_y, max_y)
            if cv2.pointPolygonTest(face_points, (float(x), float(y)), False) >= 0:
                points.append([x, y])
        
        if len(points) > 3:
            rect = (0, 0, w, h)
            subdiv = cv2.Subdiv2D(rect)
            
            for p in points:
                subdiv.insert(tuple(p))
            
            triangles = subdiv.getTriangleList()
            overlay = result.copy()
            
            for t in triangles:
                pt1 = (int(t[0]), int(t[1]))
                pt2 = (int(t[2]), int(t[3]))
                pt3 = (int(t[4]), int(t[5]))
                triangle = np.array([pt1, pt2, pt3])
                
                mask_tri = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_tri, [triangle], 255)
                
                if cv2.countNonZero(mask_tri) > 0:
                    mean_color = cv2.mean(result, mask_tri)[:3]
                    brightness = np.random.uniform(0.7, 1.3)
                    facet_color_array = np.clip(np.array(mean_color) * brightness, 0, 255)
                    facet_color = (int(facet_color_array[0]), int(facet_color_array[1]), int(facet_color_array[2]))
                    cv2.fillPoly(overlay, [triangle], facet_color)
                    cv2.polylines(overlay, [triangle], True, (255, 255, 255), 1, cv2.LINE_AA)
            
            mask_3ch = np.stack([mask] * 3, axis=2) / 255.0
            result = (overlay * mask_3ch + result * (1 - mask_3ch)).astype(np.uint8)
            result = cv2.addWeighted(result, 0.9, cv2.GaussianBlur(result, (5, 5), 0), 0.1, 20)
        
        return result
    
    # --- LIQUID METAL CORRIGE ---
    def effect_liquid_metal(self, image, landmarks, mask):
        """Texture metal liquide style T-1000."""
        h, w = image.shape[:2]
        
        # 1. Creer l'effet metallique pour toute l'image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        metal = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
        
        time_offset = int(time.time() * 10) % 360
        
        y_coords, x_coords = np.ogrid[:h, :w]
        wave_pattern_x = np.sin((x_coords + time_offset * 2) * 0.05) * 30
        wave_pattern_y = np.cos((y_coords + time_offset) * 0.03) * 30
        
        wave_pattern = wave_pattern_x + wave_pattern_y
        wave_pattern = wave_pattern.astype(np.uint8)
        
        wave_pattern_bgr = wave_pattern[..., np.newaxis].repeat(3, axis=2)
        metal = cv2.add(metal, wave_pattern_bgr)
        
        # 2. Si un visage est detecte, n'appliquer l'effet que sur le visage
        if mask is not None:
            mask_3ch = np.stack([mask] * 3, axis=2) / 255.0
            result = (metal * mask_3ch + image * (1 - mask_3ch)).astype(np.uint8)
        else:
            # 3. Si aucun visage n'est detecte, appliquer a toute l'image
            result = metal.copy()
        
        highlights = cv2.GaussianBlur(result, (5, 5), 0)
        result = cv2.addWeighted(result, 1.2, highlights, -0.2, 0)
        
        return result
    
    
    # =========================================================================
    # NOUVEAUX EFFETS (Artistique, Deformation, Overlay, Expression)
    # =========================================================================

    def effect_cartoon(self, image):
        """Effet style dessin anime / peinture."""
        return cv2.stylization(image, sigma_s=150, sigma_r=0.25)

    def effect_glitch(self, image):
        """Effet de distorsion TV (glitch) ameliore."""
        h, w = image.shape[:2]
        result = image.copy()
        
        self.glitch_counter = (self.glitch_counter + 1) % 360
        offset = int(10 * np.sin(self.glitch_counter * 0.1) + np.random.randint(-3, 3))
        
        result[:, :, 2] = np.roll(image[:, :, 2], offset, axis=1)
        result[:, :, 0] = np.roll(image[:, :, 0], -offset, axis=1)
        
        if self.glitch_counter % 5 < 2:
            num_blocks = np.random.randint(5, 15)
            max_h, max_w = h // 10, w // 10
            if max_h <= 5 or max_w <= 5: return result

            for _ in range(num_blocks):
                try:
                    y1 = np.random.randint(0, h - max_h)
                    x1 = np.random.randint(0, w - max_w)
                    bh = np.random.randint(5, max_h)
                    bw = np.random.randint(5, max_w)
                    y2, x2 = y1 + bh, x1 + bw
                    
                    offset_x = np.random.randint(-w // 10, w // 10)
                    offset_y = np.random.randint(-h // 10, h // 10)
                    
                    x1_dst, y1_dst = (x1 + offset_x) % w, (y1 + offset_y) % h
                    x2_dst, y2_dst = (x1_dst + bw), (y1_dst + bh)
                    
                    if y2_dst > h or x2_dst > w:
                        crop_h = min(bh, h - y1_dst)
                        crop_w = min(bw, w - x1_dst)
                        if crop_h > 0 and crop_w > 0:
                            result[y1_dst:y1_dst+crop_h, x1_dst:x1_dst+crop_w] = image[y1:y1+crop_h, x1:x1+crop_w]
                    else:
                        result[y1_dst:y2_dst, x1_dst:x2_dst] = image[y1:y2, x1:x2]
                        
                except Exception:
                    pass

        num_lines = np.random.randint(1, 6)
        for _ in range(num_lines):
            y = np.random.randint(0, h - 3)
            noise = np.random.randint(50, 100, (3, w, 3), dtype=np.uint8)
            result[y:y+3, :] = cv2.addWeighted(result[y:y+3, :], 0.6, noise, 0.4, 0)

        return result
            
    def effect_bulge(self, image, landmarks):
        """Deformation creative (Bulge / Fisheye) centree sur le nez."""
        h, w = image.shape[:2]
        
        center_x = int(landmarks[1].x * w)
        center_y = int(landmarks[1].y * h)
        
        K = 0.00015
        map_x, map_y = np.indices((h, w), dtype=np.float32)
        
        dx = map_x - center_y
        dy = map_y - center_x
        r_squared = dx*dx + dy*dy
        
        map_x = (center_y + dx * (1 + K * r_squared)).astype(np.float32)
        map_y = (center_x + dy * (1 + K * r_squared)).astype(np.float32)
        
        return cv2.remap(image, map_y, map_x, cv2.INTER_LINEAR)

    def effect_graphic_overlay(self, image, mask):
        """Surimpression d'une texture graphique sur le visage."""
        h, w = image.shape[:2]
        
        texture_resized = cv2.resize(self.overlay_texture, (w, h))
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        alpha = 0.4
        result = (texture_resized * mask_3ch * alpha + image * (1 - mask_3ch * alpha)).astype(np.uint8)
        
        return result

    def effect_teeth_whitening(self, image, landmarks):
        """Effet dynamique : blanchit les dents si l'utilisateur sourit bouche ouverte."""
        result = image.copy()
        
        if not self.smile_detected or not self.mouth_open:
            return result
        
        h, w = image.shape[:2]
        
        try:
            inner_mouth_mask = np.zeros((h, w), dtype=np.uint8)
            inner_mouth_points = np.array([
                [int(landmarks[i].x * w), int(landmarks[i].y * h)]
                for i in self.face_regions['inner_mouth']
            ])
            cv2.fillPoly(inner_mouth_mask, [inner_mouth_points], 255)

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            teeth_color_mask = cv2.inRange(hsv, (0, 0, 150), (179, 60, 255))
            
            final_teeth_mask = cv2.bitwise_and(inner_mouth_mask, teeth_color_mask)
            
            hsv_whitened = hsv.astype(np.float32)
            
            hsv_whitened[:, :, 1] = hsv_whitened[:, :, 1] * 0.5
            hsv_whitened[:, :, 2] = hsv_whitened[:, :, 2] * 1.2
            hsv_whitened = np.clip(hsv_whitened, 0, 255)
            
            whitened_bgr = cv2.cvtColor(hsv_whitened.astype(np.uint8), cv2.COLOR_HSV2BGR)

            final_teeth_mask_3ch = cv2.cvtColor(final_teeth_mask, cv2.COLOR_GRAY2BGR) / 255.0
            
            result = (whitened_bgr * final_teeth_mask_3ch + image * (1 - final_teeth_mask_3ch)).astype(np.uint8)
            
            return result

        except Exception:
            return image
            
    # --- SLEEPY CORRIGE ---
    def effect_sleepy(self, image):
        """Effet dynamique : flou si les yeux sont fermes."""
        if self.eyes_closed:
            # Flou tres fort si les yeux sont fermes
            return cv2.GaussianBlur(image, (61, 61), 0)
        else:
            # Aucun effet si les yeux sont ouverts
            return image


    # =========================================================================
    # TRAITEMENT PRINCIPAL (Mis a jour)
    # =========================================================================

    def process_frame(self, frame):
        """Traite la frame avec l'effet selectionne."""
        start_time = time.time()
        
        h, w = frame.shape[:2]
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        detection = self.face_mesh.process(rgb)
        rgb.flags.writeable = True
        
        landmarks = None
        mask = None
        self.face_detected = False
        
        if detection.multi_face_landmarks:
            face_landmarks_list = detection.multi_face_landmarks
            if face_landmarks_list:
                self.face_detected = True
                landmarks_raw = face_landmarks_list[0].landmark
                landmarks = self.smooth_landmarks(landmarks_raw)
                
                self.calculate_expression_metrics(landmarks, w, h)
                
                self.detection_confidence = min(1.0, self.detection_confidence + 0.2) if landmarks else 0.0
                
                if landmarks:
                    mask = self.create_face_mask(frame, landmarks)
            else:
                self.detection_confidence = max(0.0, self.detection_confidence - 0.1)
                self.calculate_expression_metrics(None, w, h)
                
        else:
            self.detection_confidence = max(0.0, self.detection_confidence - 0.1)
            self.last_landmarks = None
            self.calculate_expression_metrics(None, w, h)

        effect_name = self.effects[self.current_effect]["name"]
        
        # Application de l'effet
        # --- CORRECTION : Ajout de "Liquid Metal" et "Sleepy" a la liste d'exceptions ---
        if effect_name == "Normal" or (not self.face_detected and effect_name not in 
                                       ["Color Grading", "Glitch", "Cartoon", "Vintage Film", "Liquid Metal", "Sleepy"]):
            result = frame.copy()
        elif effect_name == "Glow Skin":
            result = self.effect_glow_skin(frame, landmarks, mask)
        elif effect_name == "Anime Eyes":
            result = self.effect_anime_eyes(frame, landmarks)
        elif effect_name == "Color Grading":
            result = self.effect_color_grading(frame)
        elif effect_name == "Portrait Mode":
            result = self.effect_portrait_mode(frame, mask)
        elif effect_name == "Beauty Filter":
            result = self.effect_beauty_filter(frame, landmarks, mask)
        elif effect_name == "Cyberpunk":
            result = self.effect_cyberpunk(frame, landmarks)
        elif effect_name == "Vintage Film":
            result = self.effect_vintage_film(frame)
        elif effect_name == "3D Depth":
            result = self.effect_3d_depth(frame, landmarks)
        elif effect_name == "Rainbow Prism":
            result = self.effect_rainbow_prism(frame, landmarks)
        elif effect_name == "Crystal Face":
            result = self.effect_crystal_face(frame, landmarks)
        # --- CORRECTION : Appel mis a jour ---
        elif effect_name == "Liquid Metal":
            result = self.effect_liquid_metal(frame, landmarks, mask)
        elif effect_name == "Cartoon":
            result = self.effect_cartoon(frame)
        elif effect_name == "Glitch":
            result = self.effect_glitch(frame)
        elif effect_name == "Bulge":
            result = self.effect_bulge(frame, landmarks)
        elif effect_name == "Overlay":
            result = self.effect_graphic_overlay(frame, mask)
        elif effect_name == "Teeth Whiten":
            result = self.effect_teeth_whitening(frame, landmarks)
        elif effect_name == "Sleepy":
            result = self.effect_sleepy(frame)
            
        else:
            result = frame.copy()

        # Calcul FPS
        fps = 1.0 / (time.time() - start_time) if time.time() - start_time > 0 else 0
        self.fps_history.append(fps)
        
        return result

    def draw_modern_ui(self, frame):
        """Interface utilisateur moderne avec couleurs Bleu Ciel et Jaune."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Definition des couleurs (BGR)
        CLR_BLEU_CIEL = (255, 191, 0)
        CLR_BLEU_FONCE = (139, 115, 0)
        CLR_BLEU_TRES_FONCE = (70, 60, 0)
        CLR_JAUNE_VIF = (0, 255, 255)
        CLR_JAUNE_FONCE = (0, 204, 204)
        CLR_ORANGE = (0, 165, 255)
        CLR_BLANC = (255, 255, 255)
        CLR_NOIR = (0, 0, 0)
        CLR_GRIS_CLAIR = (200, 200, 200)

        # ===== HEADER AVEC DETECTION ===== #
        header_h = 100
        header = np.zeros((header_h, w, 3), dtype=np.uint8)
        for i in range(header_h):
            alpha = i / header_h
            header[i] = (int(CLR_BLEU_FONCE[0] * (1 - alpha)), 
                         int(CLR_BLEU_FONCE[1] * (1 - alpha)), 
                         int(CLR_BLEU_FONCE[2] * (1 - alpha)))
        overlay[:header_h] = cv2.addWeighted(overlay[:header_h], 0.3, header, 0.7, 0)

        # Status de detection
        status_x = w - 280
        status_y = 20
        status_w = 260
        status_h = 60
        if self.face_detected:
            status_color = CLR_JAUNE_VIF
            status_bg = CLR_BLEU_FONCE
            status_text = "VISAGE DETECTE"
            confidence_text = f"{int(self.detection_confidence * 100)}%"
        else:
            status_color = CLR_JAUNE_FONCE
            status_bg = CLR_BLEU_TRES_FONCE
            status_text = "POSITIONNEZ-VOUS"
            confidence_text = "0%"

        cv2.rectangle(overlay, (status_x, status_y), (status_x + status_w, status_y + status_h), status_bg, -1)
        cv2.rectangle(overlay, (status_x, status_y), (status_x + status_w, status_y + status_h), status_color, 3)
        cv2.putText(overlay, status_text, (status_x + 15, status_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 2, cv2.LINE_AA)
        cv2.putText(overlay, f"Confiance: {confidence_text}", (status_x + 15, status_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_GRIS_CLAIR, 1, cv2.LINE_AA)

        # --- NOUVEAU : Status des expressions ---
        expression_text = ""
        if self.smile_detected and self.mouth_open:
            expression_text = "SMILE!" # Pour l'effet dents
        elif self.smile_detected:
             expression_text = "SOURIRE"
        elif self.eyes_closed:
            expression_text = "Zzz..."
            
        if expression_text:
            text_size = cv2.getTextSize(expression_text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)[0]
            text_x = status_x - text_size[0] - 20
            cv2.putText(overlay, expression_text, (text_x, status_y + 45), cv2.FONT_HERSHEY_DUPLEX, 0.9, CLR_JAUNE_VIF, 2, cv2.LINE_AA)
        
        # --- NOUVEAU NOM DE LOGICIEL ---
        title = "SmartGlace"
        cv2.putText(overlay, title, (20, 55), cv2.FONT_HERSHEY_DUPLEX, 1.2, CLR_BLEU_CIEL, 3, cv2.LINE_AA)
        
        # FPS
        if self.fps_history:
            avg_fps = np.mean(self.fps_history)
            fps_color = CLR_JAUNE_VIF if avg_fps > 20 else CLR_JAUNE_FONCE if avg_fps > 10 else CLR_ORANGE
            cv2.putText(overlay, f"FPS: {int(avg_fps)}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2, cv2.LINE_AA)

        # ===== FOOTER AVEC EFFETS =====
        footer_h = 140
        footer_y = h - footer_h
        
        footer = np.zeros((footer_h, w, 3), dtype=np.uint8)
        for i in range(footer_h):
            alpha = i / footer_h
            footer[i] = (int(CLR_BLEU_FONCE[0] * alpha), 
                         int(CLR_BLEU_FONCE[1] * alpha), 
                         int(CLR_BLEU_FONCE[2] * alpha))
        overlay[footer_y:] = cv2.addWeighted(overlay[footer_y:], 0.3, footer, 0.7, 0)

        # Carrousel d'effets
        num_visible = min(5, len(self.effects))
        card_w = 160
        card_h = 90
        card_spacing = 10
        total_width = num_visible * (card_w + card_spacing) - card_spacing
        start_x = (w - total_width) // 2

        visible_start = max(0, self.current_effect - (num_visible // 2))
        visible_end = min(len(self.effects), visible_start + num_visible)
        visible_start = max(0, visible_end - num_visible)

        for i in range(visible_start, visible_end):
            effect = self.effects[i]
            is_current = (i == self.current_effect)
            x = start_x + (i - visible_start) * (card_w + card_spacing)
            y = footer_y + 25

            if is_current:
                card_bg = CLR_JAUNE_FONCE
                card_border = CLR_JAUNE_VIF
                border_thickness = 4
                card_scale = 1.0
                text_color = CLR_NOIR
            else:
                card_bg = CLR_BLEU_FONCE
                card_border = CLR_BLEU_CIEL
                border_thickness = 2
                card_scale = 0.9
                text_color = CLR_BLANC

            scaled_w = int(card_w * card_scale)
            scaled_h = int(card_h * card_scale)
            offset_x = (card_w - scaled_w) // 2
            offset_y = (card_h - scaled_h) // 2

            cv2.rectangle(overlay, (x + offset_x, y + offset_y), (x + offset_x + scaled_w, y + offset_y + scaled_h), card_bg, -1)
            cv2.rectangle(overlay, (x + offset_x, y + offset_y), (x + offset_x + scaled_w, y + offset_y + scaled_h), card_border, border_thickness)

            icon_size = 1.0 if is_current else 0.8
            icon_text = effect["icon"]
            text_w_icon = cv2.getTextSize(icon_text, cv2.FONT_HERSHEY_SIMPLEX, icon_size, 2)[0][0]
            cv2.putText(overlay, icon_text, (x + (card_w - text_w_icon) // 2, y + 40 + offset_y), cv2.FONT_HERSHEY_SIMPLEX, icon_size, text_color, 2, cv2.LINE_AA)

            name_size = 0.5 if is_current else 0.4
            text_width = cv2.getTextSize(effect["name"], cv2.FONT_HERSHEY_SIMPLEX, name_size, 1)[0][0]
            cv2.putText(overlay, effect["name"], (x + (card_w - text_width) // 2, y + 70 + offset_y), cv2.FONT_HERSHEY_SIMPLEX, name_size, text_color, 1, cv2.LINE_AA)

        # Description de l'effet actuel
        desc_text = self.effects[self.current_effect]["description"]
        desc_width = cv2.getTextSize(desc_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
        desc_x = (w - desc_width) // 2
        cv2.rectangle(overlay, (desc_x - 20, footer_y + 122), (desc_x + desc_width + 20, footer_y + 135), CLR_NOIR, -1)
        cv2.putText(overlay, desc_text, (desc_x, footer_y + 132), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_BLEU_CIEL, 1, cv2.LINE_AA)

        # Instructions de controle
        controls = [("A/<-", "Prec"), ("D/->", "Suiv"), ("S", "Capture"), ("Q", "Quitter")]
        control_y = footer_y + 8
        control_x = 20
        for key, action in controls:
            key_w = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 20
            cv2.rectangle(overlay, (control_x, control_y), (control_x + key_w, control_y + 25), CLR_BLEU_FONCE, -1)
            cv2.rectangle(overlay, (control_x, control_y), (control_x + key_w, control_y + 25), CLR_BLEU_CIEL, 2)
            cv2.putText(overlay, key, (control_x + 10, control_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_JAUNE_VIF, 1, cv2.LINE_AA)
            cv2.putText(overlay, action, (control_x + key_w + 10, control_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_BLEU_CIEL, 1, cv2.LINE_AA)
            control_x += key_w + cv2.getTextSize(action, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 30
            
        return overlay

    # =========================================================================
    # PROGRAMME PRINCIPAL
    # =========================================================================

def main():
    print("\n" + "="*70)
    print("║" + " "*68 + "║")
    print("║" + " SMARTGLACE - MIROIR ARTISTIQUE INTELLIGENT ".center(68) + "║")
    print("║" + " "*68 + "║")
    print("="*70)
    print("Initialisation...")

    app = SmartGlaceApp()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    window_name = "SmartGlace (Bleu Ciel / Jaune)"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    screenshot_idx = 0
    
    print("\nLancement... Appuyez sur 'Q' pour quitter.")
    print("Controles: [A/<-] Effet precedent | [D/->] Effet suivant | [S] Capture")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur: Impossible de lire la frame.")
            break
            
        frame = cv2.flip(frame, 1)

        # Traitement de l'effet
        processed = app.process_frame(frame)
        
        # Interface moderne
        output = app.draw_modern_ui(processed)
        
        cv2.imshow(window_name, output)
        
        # Gestion des touches
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # Q ou ESC
            print("\nFermeture de l'application...")
            break
            
        elif key == ord('a') or key == 81 or key == 2:  # A ou fleche gauche
            app.current_effect = (app.current_effect - 1) % len(app.effects)
            effect = app.effects[app.current_effect]
            print(f"<- [{effect['icon']}] {effect['name']}")
            
        elif key == ord('d') or key == 83 or key == 3:  # D ou fleche droite
            app.current_effect = (app.current_effect + 1) % len(app.effects)
            effect = app.effects[app.current_effect]
            print(f"-> [{effect['icon']}] {effect['name']}")
            
        elif key == ord('s'):  # Screenshot
            screenshot_idx += 1
            filename = f"smart_glace_capture_{screenshot_idx:03d}.png"
            cv2.imwrite(filename, processed)
            print(f"Capture d'ecran SmartGlace sauvegardee: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("Termine.")

if __name__ == "__main__":
    main()
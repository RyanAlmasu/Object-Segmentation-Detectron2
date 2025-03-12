import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import time
from collections import defaultdict
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from sort import Sort  
import tempfile

# -------------------- CSS STYLE --------------------
st.markdown("""
<style>
    .stApp { padding: 2rem; }
    .header { color: #2F4F4F; border-bottom: 3px solid #2F4F4F; padding-bottom: 0.5rem; }
    .sidebar .sidebar-content { background-color: #F0F2F6; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; padding: 0.5rem 1rem; }
    .stDownloadButton>button { background-color: #008CBA; color: white; }
</style>
""", unsafe_allow_html=True)

# -------------------- SETUP MODEL --------------------
@st.cache_resource
def setup_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device
    return DefaultPredictor(cfg), MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), device

predictor, metadata, device_used = setup_model()

# -------------------- SEGMENTASI GAMBAR --------------------
def segment_image(image_np, predictor, metadata, confidence_threshold):
    outputs = predictor(image_np)
    instances = outputs["instances"]
    filtered_instances = instances[instances.scores >= confidence_threshold]
    
    v = Visualizer(image_np[:, :, ::-1], metadata, scale=1.2)
    out = v.draw_instance_predictions(filtered_instances.to("cpu"))
    return out.get_image(), filtered_instances

def handle_image_upload(uploaded_file, predictor, metadata, confidence_threshold):
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)[:, :, ::-1]
    
    with st.spinner('Memproses gambar...'):
        result_image, filtered_instances = segment_image(image_np, predictor, metadata, confidence_threshold)

    col1, col2 = st.columns(2)
    with col1: st.image(image, caption="Gambar Asli", use_container_width=True)
    with col2: st.image(result_image, caption="Hasil Segmentasi", use_container_width=True)

    if len(filtered_instances) > 0:
        st.markdown("### Informasi Deteksi Objek")
        st.write(f"Jumlah objek yang terdeteksi: {len(filtered_instances)}")

        class_counts = defaultdict(int)
        for pred_class in filtered_instances.pred_classes:
            class_name = metadata.thing_classes[pred_class]
            class_counts[class_name] += 1
        
        st.write("Jumlah objek per kelas:")
        for class_name, count in class_counts.items():
            st.write(f"- {class_name}: {count}")

    return result_image

# -------------------- OBJECT TRACKING (VIDEO) --------------------
def process_video(uploaded_video, predictor, confidence_threshold):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_video.read())
    temp_file.close() 

    cap = cv2.VideoCapture(temp_file.name)
    FRAME_WINDOW = st.image([])
    mot_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.3)
    trajectory_history = defaultdict(list)

    total_objects = 0
    class_counts = defaultdict(int)
    object_details = defaultdict(lambda: {"class": None, "confidence": 0, "count": 0}) 


    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        outputs = predictor(frame_rgb)
        instances = outputs["instances"]
        filtered_instances = instances[instances.scores >= confidence_threshold]

        if len(filtered_instances) > 0:
            boxes = filtered_instances.pred_boxes.tensor.cpu().numpy()
            scores = filtered_instances.scores.cpu().numpy().reshape(-1, 1)
            dets = np.hstack((boxes, scores))
        else:
            dets = np.empty((0, 5))
        
        tracked_objs = mot_tracker.update(dets)
        
        v = Visualizer(frame_rgb[:, :, ::-1], metadata, scale=1.2)
        out_frame = v.draw_instance_predictions(filtered_instances.to("cpu"))
        vis_frame = cv2.cvtColor(out_frame.get_image(), cv2.COLOR_BGR2RGB)

        for obj in tracked_objs:
            x1, y1, x2, y2, obj_id = obj
            obj_id = int(obj_id)
            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                vis_frame, 
                f"ID: {obj_id}", 
                (int(x1), int(y1)-10), 
                cv2.FONT_HERSHEY_SIMPLEX,  
                0.5,  
                (0, 255, 0),
                2  
                )

            center = (int((x1+x2)/2), int((y1+y2)/2))
            trajectory_history[obj_id].append(center)
            if len(trajectory_history[obj_id]) > 15:
                trajectory_history[obj_id].pop(0)

            points = np.array(trajectory_history[obj_id], np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_frame, [points], False, (0, 255, 50), 2)

            if len(filtered_instances) > 0:
                pred_class = filtered_instances.pred_classes[obj_id % len(filtered_instances)]
                class_name = metadata.thing_classes[pred_class]
                confidence = scores[obj_id % len(scores)][0]

                if obj_id in object_details:
                    object_details[obj_id]["count"] += 1
                else:
                    object_details[obj_id] = {
                        "class": class_name,
                        "confidence": confidence,
                        "count": 1
                    }

        total_objects = len(object_details) 
        class_counts = defaultdict(int)
        for obj_id, details in object_details.items():
            class_counts[details["class"]] += 1
        out_video.write(cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))

        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Memproses frame {frame_count} dari {total_frames} ({progress * 100:.2f}%)")

        FRAME_WINDOW.image(vis_frame)
        time.sleep(0.02)

    cap.release()
    out_video.release()
    st.success("Proses tracking selesai!")
    st.markdown("### Informasi Deteksi Objek")
    st.write(f"Total objek yang terdeteksi: {total_objects}")
    st.write("Jumlah objek per kelas:")
    for class_name, count in class_counts.items():
        st.write(f"- {class_name}: {count}")

    st.markdown("### Detail Objek yang Terdeteksi")
    st.table([
        {"ID": obj_id, **details} for obj_id, details in object_details.items()
    ])
    st.markdown("### Download Hasil Video")
    with open(output_video_path, "rb") as file:
        btn = st.download_button(
            label="Download Video Hasil Tracking",
            data=file,
            file_name="tracked_video.mp4",
            mime="video/mp4"
        )
        
# -------------------- OBJECT TRACKING (WEBCAM) --------------------
def process_webcam(predictor, confidence_threshold, frame_window):
    cap = cv2.VideoCapture(0)
    mot_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.3)
    trajectory_history = defaultdict(list)
    total_objects = 0
    class_counts = defaultdict(int)

    while st.session_state.webcam_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal mengambil frame dari webcam")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        outputs = predictor(frame_rgb)
        instances = outputs["instances"]
        filtered_instances = instances[instances.scores >= confidence_threshold]

        if len(filtered_instances) > 0:
            boxes = filtered_instances.pred_boxes.tensor.cpu().numpy()
            scores = filtered_instances.scores.cpu().numpy().reshape(-1, 1)
            dets = np.hstack((boxes, scores))
        else:
            dets = np.empty((0, 5))
        
        tracked_objs = mot_tracker.update(dets)

        v = Visualizer(frame_rgb[:, :, ::-1], metadata, scale=1.2)
        out_frame = v.draw_instance_predictions(filtered_instances.to("cpu"))
        vis_frame = cv2.cvtColor(out_frame.get_image(), cv2.COLOR_BGR2RGB)

        for obj in tracked_objs:
            x1, y1, x2, y2, obj_id = obj
            obj_id = int(obj_id)
            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                vis_frame, 
                f"ID: {obj_id}", 
                (int(x1), int(y1)-10),  
                cv2.FONT_HERSHEY_SIMPLEX,  
                0.5,  
                (0, 255, 0),  
                2  
                )

            center = (int((x1+x2)/2), int((y1+y2)/2))
            trajectory_history[obj_id].append(center)
            if len(trajectory_history[obj_id]) > 15:
                trajectory_history[obj_id].pop(0)

            points = np.array(trajectory_history[obj_id], np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_frame, [points], False, (0, 255, 50), 2)

        # menghitung jumlah objek per frame
        total_objects += len(filtered_instances)
        for pred_class in filtered_instances.pred_classes:
            class_name = metadata.thing_classes[pred_class]
            class_counts[class_name] += 1

        frame_window.image(vis_frame)
        time.sleep(0.02)

    cap.release()
    st.session_state.webcam_active = False
    st.success("Webcam dihentikan")
    st.markdown("### Informasi Deteksi Objek")
    st.write(f"Total objek yang terdeteksi: {total_objects}")
    st.write("Jumlah objek per kelas:")
    for class_name, count in class_counts.items():
        st.write(f"- {class_name}: {count}")

# -------------------- MAIN STREAMLIT APP --------------------
def main():
    st.markdown("<h1 class='header'>Object Detection & Tracking</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## Pengaturan")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        st.markdown("## Informasi Perangkat")
        st.write(f"Model berjalan di: {'GPU' if device_used == 'cuda' else 'CPU'}")
        st.write(f"Model yang digunakan: COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

    tab1, tab2, tab3 = st.tabs(["Upload Gambar", "Upload Video", "Realtime Webcam"])

    with tab1:
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            handle_image_upload(uploaded_file, predictor, metadata, confidence_threshold)

    with tab2:
        uploaded_video = st.file_uploader("Upload Video...", type=["mp4", "avi", "mov"])
        if uploaded_video:
            if st.button("Mulai Tracking Video"):
                process_video(uploaded_video, predictor, confidence_threshold)

    with tab3:
        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False
            
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Mulai Webcam") and not st.session_state.webcam_active:
                st.session_state.webcam_active = True
        with col2:
            if st.button("Hentikan Webcam") and st.session_state.webcam_active:
                st.session_state.webcam_active = False

        FRAME_WINDOW = st.empty()
        
        if st.session_state.webcam_active:
            process_webcam(predictor, confidence_threshold, FRAME_WINDOW)

if __name__ == "__main__":
    main()
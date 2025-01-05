import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import mediapipe as mp
import cv2
import pickle  # モデルの保存・読み込みに使用
import math
import matplotlib.pyplot as plt
import streamlit as st
import io
import zipfile

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# スケーリング用のオブジェクトを用意
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

def get_cheek_landmarks(face_landmarks, h, w):
    """
    Mediapipeのランドマークから頬のランドマークポイントを取得し、
    ピクセル座標（整数値）として返します。
    """

    # 左頬のランドマークポイントをピクセル座標に変換
    left_cheek = [
        (int(face_landmarks.landmark[234].x * w), int(face_landmarks.landmark[234].y * h)),  # 左頬の外側
        (int(face_landmarks.landmark[138].x * w), int(face_landmarks.landmark[138].y * h)),  # 左頬の中心寄り
        (int(face_landmarks.landmark[185].x * w), int(face_landmarks.landmark[185].y * h)),  # 上唇の左側近辺
        (int(face_landmarks.landmark[232].x * w), int(face_landmarks.landmark[232].y * h))   # 左目の外側近辺
    ]

    # 右頬のランドマークポイントをピクセル座標に変換
    right_cheek = [
        (int(face_landmarks.landmark[454].x * w), int(face_landmarks.landmark[454].y * h)),  # 右頬の外側
        (int(face_landmarks.landmark[367].x * w), int(face_landmarks.landmark[367].y * h)),  # 右頬の中心寄り
        (int(face_landmarks.landmark[409].x * w), int(face_landmarks.landmark[409].y * h)),  # 上唇の右側近辺
        (int(face_landmarks.landmark[452].x * w), int(face_landmarks.landmark[452].y * h))   # 右目の外側近辺
    ]

    return left_cheek, right_cheek

def extract_cheek_region(image, cheek_coords):
    """頬領域のマスクを作成し、該当部分のRGB値を抽出します。"""
    pil_image = Image.fromarray(image)
    mask = Image.new('L', pil_image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(cheek_coords, outline=1, fill=1)
    mask = np.array(mask)
    cheek_pixels = np.array(pil_image)[:, :, :3]
    cheek_rgb = cheek_pixels[mask == 1]
    return cheek_rgb

def calculate_brightness(rgb_values):
    """明るさを計算します。"""
    return rgb_values.mean()

def calculate_contrast(rgb_values):
    """コントラストを計算します。"""
    luminance = 0.2126 * rgb_values[:, 0] + 0.7152 * rgb_values[:, 1] + 0.0722 * rgb_values[:, 2]
    contrast = luminance.max() - luminance.min()
    return contrast

def calculate_saturation(rgb_values):
    """彩度を計算します。"""
    max_rgb = rgb_values.max(axis=1)
    min_rgb = rgb_values.min(axis=1)
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-5)  # 過剰分散を防ぐ
    return saturation.mean()

def get_landmarks(image):
    """Mediapipeを使用してポーズと顔のランドマークを取得します。"""
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB形式に変換
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:

        pose_results = pose.process(image_rgb)
        face_results = face_mesh.process(image_rgb)

        if pose_results.pose_landmarks and face_results.multi_face_landmarks:
            pose_landmarks = pose_results.pose_landmarks.landmark
            face_landmarks = face_results.multi_face_landmarks[0].landmark

            def get_face_point(index):
                landmark = face_landmarks[index]
                return int(landmark.x * w), int(landmark.y * h)

            return {
                'pose': pose_landmarks,
                'face': {
                    'right_forehead': get_face_point(103),  # 額右
                    'left_forehead': get_face_point(332),   # 額左
                    'right_cheek': get_face_point(93),      # 頬右
                    'left_cheek': get_face_point(352),      # 頬左
                    'right_jaw': get_face_point(172),       # 顎右
                    'left_jaw': get_face_point(397)         # 顎左
                }
            }
    print("ランドマークが検出されませんでした。")
    return None

def extract_body_region(image, pose_landmarks, h, w):
    """10点（額右、頬右、顎右、右肩、右ひじ、左ひじ、左肩、顎左、頬左、額左）からなる体の領域を取得します。"""
    pose_points = pose_landmarks['pose']  # Mediapipeから取得したポーズランドマーク
    face_points = pose_landmarks['face']  # Mediapipeから取得した顔ランドマーク

    try:
        body_points = [
            face_points['right_forehead'],  # 額右
            face_points['right_cheek'],    # 頬右
            face_points['right_jaw'],      # 顎右
            (pose_points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
             pose_points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h),  # 右肩
            (pose_points[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
             pose_points[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h),     # 右ひじ
            (pose_points[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
             pose_points[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h),      # 左ひじ
            (pose_points[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
             pose_points[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h),   # 左肩
            face_points['left_jaw'],       # 顎左
            face_points['left_cheek'],     # 頬左
            face_points['left_forehead']   # 額左
        ]
        
        # 各座標を整数に変換
        body_points = [(int(x), int(y)) for x, y in body_points]

    except Exception as e:
        print(f"Error in extracting body points: {e}")
        raise

    return np.array(body_points, np.int32)

def extract_body_features(image, body_coords, output_path="output_body_region.jpg"):
    """体領域の特性を抽出します。"""
    # 体領域を描画
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # OpenCVはBGR形式なのでRGBに変換
    body_coords = [(int(x), int(y)) for x, y in body_coords]  # 座標を整数化

    mask = Image.new('L', pil_image.size, 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.polygon(body_coords, outline=1, fill=1)

    mask = np.array(mask)
    body_pixels = np.array(pil_image)[:, :, :3]
    body_rgb = body_pixels[mask == 1]
    return body_rgb

def calculate_image_brightness(image):
    """画像全体の明るさを計算します。"""
    rgb_values = image.reshape(-1, 3)
    return calculate_brightness(rgb_values)

def apply_coefficients_multivariate_with_scalers(test_image, models, scaler_X, scaler_Y):
    """
    テスト画像に対してモデルで予測されたRGB係数を適用します。
    """
    h, w, _ = test_image.shape
    #print("Getting landmarks...")
    pose_landmarks = get_landmarks(test_image)

    if not pose_landmarks:
        raise ValueError("体のランドマークが検出されませんでした。")

    # Attempt to extract body coordinate
    body_coords = extract_body_region(test_image, pose_landmarks, h, w)
    body_coords = [(int(x), int(y)) for x, y in body_coords]  # 座標を整数化

    if body_coords is None:
        raise ValueError("Body coordinates could not be determined.")
        
    body_rgb = extract_body_features(test_image, body_coords, output_path="output_body_region_red.jpg")

    body_contrast = calculate_contrast(body_rgb)
    body_saturation = calculate_saturation(body_rgb)

    # 画像全体の明るさを計算
    image_brightness = calculate_image_brightness(test_image)

    # 頬の明るさも同時に計算
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        before_results = face_mesh.process(test_image)
        if before_results.multi_face_landmarks:
            for face_landmarks in before_results.multi_face_landmarks:
                before_left_cheek, before_right_cheek = get_cheek_landmarks(face_landmarks, h, w)

    left_cheek_rgb = extract_cheek_region(test_image, before_left_cheek)
    right_cheek_rgb = extract_cheek_region(test_image, before_right_cheek)
    cheek_brightness = calculate_brightness(np.vstack([left_cheek_rgb, right_cheek_rgb]))

    before_left_rgb = extract_cheek_region(test_image, before_left_cheek)
    before_right_rgb = extract_cheek_region(test_image, before_right_cheek)

    before_avg_rgb = np.mean([before_left_rgb.mean(axis=0), before_right_rgb.mean(axis=0)], axis=0)

    # 特徴量ベクトルをスケーリング
    feature_vector = scaler_X.transform([[image_brightness, cheek_brightness, body_saturation, body_contrast]])

    # RGB係数の予測
    coeff_r_scaled = models[0].predict(feature_vector)[0]
    coeff_g_scaled = models[1].predict(feature_vector)[0]
    coeff_b_scaled = models[2].predict(feature_vector)[0]

    # スケールを元に戻す
    coeffs_scaled = np.array([[coeff_r_scaled, coeff_g_scaled, coeff_b_scaled]])
    coeffs = scaler_Y.inverse_transform(coeffs_scaled)[0]

    print(f"Predicted coefficients: R={coeffs[0]}, G={coeffs[1]}, B={coeffs[2]}")
 
    # 調整
    predicted_coeff = np.clip(coeffs, 0, 255)
    adjusted_image = np.clip(test_image * predicted_coeff, 0, 255).astype(np.uint8)

    return adjusted_image

def load_model_with_scalers(uploaded_file):
    # UploadedFile から直接読み込む
    models = pickle.load(uploaded_file)

    if not all(key in models for key in ['model_r', 'model_g', 'model_b', 'scaler_X', 'scaler_Y']):
        raise ValueError("保存されたデータに必要なキーが含まれていません。")
    
    print("モデルを読み込みました")
    return models['model_r'], models['model_g'], models['model_b'], models['scaler_X'], models['scaler_Y']


# タイトルと説明
st.title("レタッチツール（仮） アウトプット版")
st.write("学習モデルを使用して画像のRGB特性を自動調整します。")

# 初期化
if 'downloaded' not in st.session_state:
    st.session_state.downloaded = False
    
def clear_uploads():
    st.session_state.processed = False
    st.session_state.downloaded = False
    if 'uploaded_model_file' in st.session_state:
        del st.session_state.uploaded_model_file
    if 'uploaded_files' in st.session_state:
        del st.session_state.uploaded_files
        
# アップロードモデルファイルの初期化
uploaded_model_file = []

if not st.session_state.downloaded:
    # モデルのロード
    uploaded_model_file = st.file_uploader("モデルファイルをアップロードしてください", type=["pkl"])

if uploaded_model_file is not None and not st.session_state.downloaded:
    model_r, model_g, model_b, scaler_X, scaler_Y = load_model_with_scalers(uploaded_model_file)
    models = (model_r, model_g, model_b)

    # ファイルアップロード
    uploaded_files = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        st.write("アップロードされたファイルの数:", len(uploaded_files))

        # 処理された画像を格納するリスト
        processed_images = []
        zip_buffer = io.BytesIO()

        # 代表画像（1枚目）を表示するためのフラグ
        first_image_displayed = False

        for uploaded_file in uploaded_files:
            # アップロードされた画像をPIL形式で読み込む
            image = Image.open(uploaded_file)
            # PIL画像をOpenCV形式に変換
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # 代表のアップロード画像を表示
            if not first_image_displayed:
                st.image(np.array(image), caption="アップロードされた画像（1枚目）", use_column_width=True)
                first_image_displayed = True

            # モデルを適用して画像を調整
            adjusted_image_cv = apply_coefficients_multivariate_with_scalers(image_cv, models, scaler_X, scaler_Y)
            output_image = Image.fromarray(cv2.cvtColor(adjusted_image_cv, cv2.COLOR_BGR2RGB))

            # バイトデータとして保存
            buffer = io.BytesIO()
            output_image.save(buffer, format="JPEG", quality=100)
            processed_images.append((uploaded_file.name, buffer.getvalue()))

        # ZIPファイルを作成
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for file_name, data in processed_images:
                zf.writestr(file_name, data)

        # 調整後の画像（1枚目）を表示
        st.image(Image.open(io.BytesIO(processed_images[0][1])), caption="調整後の画像（1枚目）", use_column_width=True)

        st.session_state.downloaded = True  # ダウンロード完了フラグを立てる
        
        # ZIPファイルをダウンロード可能にする
        st.download_button(
            label="調整後の画像を一括ダウンロード",
            data=zip_buffer.getvalue(),
            file_name="processed_images.zip",
            mime="application/zip"
        )

if uploaded_model_file == [] and st.session_state.downloaded:
    # モデルの学習処理
    st.session_state.processed = True  # 学習完了フラグを立てる
    
    # 学習完了後、クリアボタンを表示
    if st.button('アップロードをクリアして新しい学習を開始'):
        clear_uploads()
        st.rerun()
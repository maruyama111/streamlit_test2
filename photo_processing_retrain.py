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
import zipfile
import io
import sys

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

def list_files_with_correct_encoding(directory):
    """指定ディレクトリ内のファイルを正しくリストします（エンコード問題を回避）"""
    files = []
    for filename in os.listdir(directory):
        try:
            # ファイル名をエンコードしてリストに追加
            files.append(os.fsdecode(filename))
        except UnicodeDecodeError:
            print(f"UnicodeDecodeError: {filename}")
    return files

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

        #Debug: Print the extracted body points
        #print(f"10点の体領域ポイント: {body_points}")

    except Exception as e:
        print(f"Error in extracting body points: {e}")
        raise

    return np.array(body_points, np.int32)

def extract_body_features(image, body_coords, output_path="output_body_region.jpg"):
    """体領域の特性を抽出します。"""
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

def process_image_pair(before_image, after_image):
    """加工前後の画像ペアから頬の平均RGB値と明るさを抽出します。"""
    h, w, _ = before_image.shape

    #print("Getting landmarks...")
    pose_landmarks = get_landmarks(before_image)

    if not pose_landmarks:
        raise ValueError("体のランドマークが検出されませんでした。")
    try:
        # Attempt to extract body coordinate
        body_coords = extract_body_region(before_image, pose_landmarks, h, w)
        body_coords = [(int(x), int(y)) for x, y in body_coords]  # 座標を整数化

        if body_coords is None:
            raise ValueError("Body coordinates could not be determined.")
        
        # Debug: Raw coordinates
        #print(f"Raw body_coords: {body_coords}")

    except Exception as e:
        # Handle exceptions and continue processing other images
        print(f"Error processing image pair : {e}")

    #print("Body region extracted. Extracting body features...")
    body_rgb = extract_body_features(before_image, body_coords, output_path="output_body_region_red.jpg")

    body_contrast = calculate_contrast(body_rgb)
    body_saturation = calculate_saturation(body_rgb)

    # 画像全体の明るさを計算
    image_brightness = calculate_image_brightness(before_image)

    # 頬の明るさも同時に計算
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        before_results = face_mesh.process(before_image)
        if before_results.multi_face_landmarks:
            for face_landmarks in before_results.multi_face_landmarks:
                before_left_cheek, before_right_cheek = get_cheek_landmarks(face_landmarks, h, w)
        after_results = face_mesh.process(after_image)
        if after_results.multi_face_landmarks:
            for face_landmarks in after_results.multi_face_landmarks:
                after_left_cheek, after_right_cheek = get_cheek_landmarks(face_landmarks, h, w)

    left_cheek_rgb = extract_cheek_region(before_image, before_left_cheek)
    right_cheek_rgb = extract_cheek_region(before_image, before_right_cheek)
    cheek_brightness = calculate_brightness(
        np.vstack([left_cheek_rgb, right_cheek_rgb])
    )

    before_left_rgb = extract_cheek_region(before_image, before_left_cheek)
    before_right_rgb = extract_cheek_region(before_image, before_right_cheek)
    after_left_rgb = extract_cheek_region(after_image, after_left_cheek)
    after_right_rgb = extract_cheek_region(after_image, after_right_cheek)

    before_avg_rgb = np.mean([before_left_rgb.mean(axis=0), before_right_rgb.mean(axis=0)], axis=0)
    after_avg_rgb = np.mean([after_left_rgb.mean(axis=0), after_right_rgb.mean(axis=0)], axis=0)

    #print("Calculating RGB coefficients...")
    coeff = after_avg_rgb / before_avg_rgb

    # デバッグ用の出力
    print(f"Image Brightness: {image_brightness}")
    print(f"Cheek Brightness: {cheek_brightness}")
    print(f"Body Saturation: {body_saturation}")
    print(f"Body Contrast: {body_contrast}")
    print(f"Coeff: {coeff}")

    return image_brightness, cheek_brightness, body_saturation, body_contrast, coeff

def prepare_training_data(before_files, after_files):
    """学習用データセットを準備します。"""
    if len(before_files) != len(after_files):
        raise ValueError("beforeとafterの画像数が一致しません。")

    data = []
    for before_file, after_file in zip(before_files, after_files):
        try:
            # 画像ペアを読み込む
            before_image = Image.open(before_file)
            after_image = Image.open(after_file)

            # PIL画像をOpenCV形式に変換
            before_image_cv = cv2.cvtColor(np.array(before_image), cv2.COLOR_RGB2BGR)
            after_image_cv = cv2.cvtColor(np.array(after_image), cv2.COLOR_RGB2BGR)

            # 画像ペアから特徴量を計算
            result = process_image_pair(before_image_cv, after_image_cv)
            if result:
                image_brightness, cheek_brightness, body_saturation, body_contrast, coeff = result
                data.append({
                    'image_brightness': image_brightness,
                    'cheek_brightness': cheek_brightness,
                    'body_saturation': body_saturation,
                    'body_contrast': body_contrast,
                    'coeff_r': coeff[0],
                    'coeff_g': coeff[1],
                    'coeff_b': coeff[2]
                })

        except Exception as e:
            print(f"画像ペア {before_file.name} と {after_file.name} の処理中にエラーが発生しました: {e}")
            continue

    # リストをPandasのDataFrameに変換して返す
    new_data_df = pd.DataFrame(data)
    return new_data_df  # ここでDataFrameを返す

@st.cache_data
def load_model_with_scalers(uploaded_file):
    # UploadedFile から直接読み込む
    models = pickle.load(uploaded_file)

    if not all(key in models for key in ['model_r', 'model_g', 'model_b', 'scaler_X', 'scaler_Y']):
        raise ValueError("保存されたデータに必要なキーが含まれていません。")
    
    print("モデルを読み込みました")
    return models['model_r'], models['model_g'], models['model_b'], models['scaler_X'], models['scaler_Y']

def fine_tune_model_with_scalers(models, scaler_X, scaler_Y, new_data_df):
    """新しいデータを学習済みスケーリングでスケールしてモデルを追加学習"""
    model_r, model_g, model_b = models

    # 新しいデータのスケーリング
    X_new = new_data_df[['image_brightness', 'cheek_brightness', 'body_saturation', 'body_contrast']].values
    Y_new = new_data_df[['coeff_r', 'coeff_g', 'coeff_b']].values
    X_new_scaled = scaler_X.transform(X_new)
    Y_new_scaled = scaler_Y.transform(Y_new)

    # 各モデルの部分学習
    model_r.partial_fit(X_new_scaled, Y_new_scaled[:, 0])  # 赤
    model_g.partial_fit(X_new_scaled, Y_new_scaled[:, 1])  # 緑
    model_b.partial_fit(X_new_scaled, Y_new_scaled[:, 2])  # 青

    print("追加学習を完了しました。")
    return model_r, model_g, model_b

def save_model_with_scalers(models, scaler_X, scaler_Y):
    """モデルとスケーラーをバイナリデータとして返す"""
    model_buffer = io.BytesIO()
    pickle.dump({
        'model_r': models[0],
        'model_g': models[1],
        'model_b': models[2],
        'scaler_X': scaler_X,
        'scaler_Y': scaler_Y
    }, model_buffer)
    model_buffer.seek(0)  # バッファの先頭に移動
    return model_buffer.getvalue()  # バイナリデータを返す


# タイトルと説明
st.title("レタッチツール（仮） 学習版")
st.write("レタッチ済みの画像ペアを追加学習させてモデルを更新します")

# アップロードファイルのクリア
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'processed' not in st.session_state:
    st.session_state.processed = False
    
if 'downloaded' not in st.session_state:
    st.session_state.downloaded = False

def clear_uploads():
    st.session_state.uploaded_files = []
    st.session_state.processed = False
    st.session_state.downloaded = False
    if 'uploaded_model_file' in st.session_state:
        del st.session_state.uploaded_model_file
    if 'before_files' in st.session_state:
        del st.session_state.before_files
    if 'after_files' in st.session_state:
        del st.session_state.after_files

# アップロードモデルファイルの初期化
uploaded_model_file = []

if not st.session_state.downloaded:
    # モデルのロード
    uploaded_model_file = st.file_uploader("追加学習させたいモデルファイルをアップロードしてください", type=["pkl"])

if uploaded_model_file is not None and not st.session_state.downloaded:
    try:
        model_r, model_g, model_b, scaler_X, scaler_Y = load_model_with_scalers(uploaded_model_file)
        models = (model_r, model_g, model_b)
        st.write("モデルが正常にロードされました。")

        # ファイルアップロード
        before_files = st.file_uploader("追加学習させるbefore画像をアップロードしてください", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        after_files = st.file_uploader("追加学習させるafter画像をアップロードしてください", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        st.write("※before画像とafter画像のファイル名は一致させてください")

        if before_files and after_files:
            if len(before_files) != len(after_files):
                st.error("before画像とafter画像のファイル数が一致しません")
            else:
                st.success("アップロード完了！")
        
            # 学習データを準備
            st.write("新しい学習データを準備しています...")
            new_data_df = prepare_training_data(before_files, after_files)

            # モデルを更新
            st.write("モデルを追加学習中です...")
            updated_models = fine_tune_model_with_scalers(models, scaler_X, scaler_Y, new_data_df)

            # モデルを保存してバイナリデータを取得
            st.write("更新されたモデルを準備しています...")
            updated_model_data = save_model_with_scalers(updated_models, scaler_X, scaler_Y)
            st.write("モデルの更新が完了しました！")
            
            st.session_state.downloaded = True  # ダウンロード完了フラグを立てる
            
            # ダウンロードボタンを作成         
            st.download_button(
                label="更新されたモデルをダウンロード",
                data=updated_model_data,
                file_name="updated_model.pkl",
                mime="application/octet-stream"
            )

    except Exception as e:
        st.error(f"モデルのロード中にエラーが発生しました: {e}")
else:
    st.warning("モデルファイルをアップロードしてください")

if uploaded_model_file == [] and st.session_state.downloaded:
    # モデルの学習処理
    st.session_state.processed = True  # 学習完了フラグを立てる
    
    # 学習完了後、クリアボタンを表示
    if st.button('アップロードをクリアして新しい学習を開始'):
        clear_uploads()
        st.rerun()
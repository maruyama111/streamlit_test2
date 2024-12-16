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

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# スケーリング用のオブジェクトを用意
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

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

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
    return image

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

        # Debug: Print the extracted body points
        #print(f"10点の体領域ポイント: {body_points}")

    except Exception as e:
        print(f"Error in extracting body points: {e}")
        raise

    return np.array(body_points, np.int32)

def extract_body_features(image, body_coords, output_path="output_body_region.jpg"):
    """体領域の特性を抽出します。"""
    # 体領域を描画
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # OpenCVはBGR形式なのでRGBに変換
    #draw_overlay = ImageDraw.Draw(pil_image)

    # 赤線で体領域を描画（閉じるために始点を終点として追加）
    body_coords = [(int(x), int(y)) for x, y in body_coords]  # 座標を整数化
    #draw_overlay.polygon(body_coords, outline="red")  # ポリゴンとして描画

    # 保存処理
    #pil_image.save(output_path)
    #print(f"描画した画像を保存しました: {output_path}")

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

def process_image_pair(before_path, after_path):
    """加工前後の画像ペアから頬の平均RGB値と明るさを抽出します。"""
    before_image = load_image(before_path)
    after_image = load_image(after_path)

    if before_image is None or after_image is None:
        print(f"One or both images could not be loaded: {before_path}, {after_path}")
        return None
    
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
        print(f"Error processing image pair {before_path} and {after_path}: {e}")

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

def prepare_training_data(before_dir, after_dir):
    """学習用データセットを準備します。"""
    data = []
    before_dir = os.path.abspath(before_dir)  # 絶対パスに変換
    after_dir = os.path.abspath(after_dir)    # 絶対パスに変換

    # ディレクトリ内のファイルをリスト化
    before_images = list_files_with_correct_encoding(before_dir)
    after_images = list_files_with_correct_encoding(after_dir)

    if len(before_images) != len(after_images):
        raise ValueError("beforeとafterの画像数が一致しません。")

    for before_img, after_img in zip(before_images, after_images):
        before_path = os.path.join(before_dir, before_img)
        after_path = os.path.join(after_dir, after_img)

        try:
            # 画像ペアから特徴量を計算
            result = process_image_pair(before_path, after_path)
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
            print(f"画像ペア {before_img} と {after_img} の処理中にエラーが発生しました: {e}")
            continue

    # リストをPandasのDataFrameに変換して返す
    new_data_df = pd.DataFrame(data)
    return new_data_df  # ここでDataFrameを返す

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
        
    # Debug: Raw coordinates
    #print(f"Raw body_coords: {body_coords}")

    #print("Body region extracted. Extracting body features...")
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

def load_model_with_scalers(model_path='trained_4factor_model2.pkl'):
    """保存された辞書型データから各カラーチャンネルのモデルを読み込む"""
    with open(model_path, 'rb') as file:
        data = pickle.load(file)

    if not all(key in data for key in ['model_r', 'model_g', 'model_b', 'scaler_X', 'scaler_Y']):
        raise ValueError("保存されたデータに必要なキーが含まれていません。")
    
    print(f"モデルを読み込みました: {model_path}")
    return (data['model_r'], data['model_g'], data['model_b'], 
            data['scaler_X'], data['scaler_Y'])

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

def save_model_with_scalers(models, scaler_X, scaler_Y, model_path='updated_4factor_model2.pkl'):
    """モデルとスケーラーを保存する"""
    with open(model_path, 'wb') as file:
        pickle.dump({
            'model_r': models[0],
            'model_g': models[1],
            'model_b': models[2],
            'scaler_X': scaler_X,
            'scaler_Y': scaler_Y
        }, file)
    print(f"モデルとスケール係数を保存しました: {model_path}")

# モデルのロード
model_path = 'trained_4factor_model3.pkl'
model_r, model_g, model_b, scaler_X, scaler_Y = load_model_with_scalers(model_path)
models = (model_r, model_g, model_b)


# 新しいデータで追加学習
#new_data_dir_before = 'new_data/before'
#new_data_dir_after = 'new_data/after'
#new_data_df = prepare_training_data(new_data_dir_before, new_data_dir_after)

# モデルを更新
#models = (model_r, model_g, model_b)
#updated_models = fine_tune_model_with_scalers(models, scaler_X, scaler_Y, new_data_df)

# 更新されたモデルとスケール係数を保存
#updated_model_path = 'updated_4factor_model3.pkl'
#save_model_with_scalers(updated_models, scaler_X, scaler_Y, updated_model_path)

# 新しい画像に適用
#test_image_path = 'test_images/test1.jpg'  # 新しい画像のパス
#output_image_path = 'output_images/adjusted_test1_4factor_3.jpg'  # 出力画像のパス
#test_image = load_image(test_image_path)

#adjusted_image = apply_coefficients_multivariate_with_scalers(test_image, updated_models, scaler_X, scaler_Y)

# 調整後の画像を保存
#output_image = Image.fromarray(cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2BGR))
#output_image.save(output_image_path)
#print(f"調整後の画像を保存しました: {output_image_path}")


# タイトルと説明
st.title("レタッチツール（仮）")
st.write("学習モデルを使用して画像のRGB特性を自動調整します。")

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
        output_image.save(buffer, format="JPEG")
        processed_images.append((uploaded_file.name, buffer.getvalue()))

    # ZIPファイルを作成
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for file_name, data in processed_images:
            zf.writestr(file_name, data)

    # 調整後の画像（1枚目）を表示
    st.image(Image.open(io.BytesIO(processed_images[0][1])), caption="調整後の画像（1枚目）", use_column_width=True)

    # ZIPファイルをダウンロード可能にする
    st.download_button(
        label="調整後の画像を一括ダウンロード",
        data=zip_buffer.getvalue(),
        file_name="processed_images.zip",
        mime="application/zip"
    )
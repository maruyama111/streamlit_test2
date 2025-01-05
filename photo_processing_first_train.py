import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import mediapipe as mp
import cv2
import pickle  # モデルの保存・読み込みに使用

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
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-5)
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

def process_image_pair(before_path, after_path):
    """加工前後の画像ペアから特性を抽出します。"""
    before_image = load_image(before_path)
    after_image = load_image(after_path)

    if before_image is None or after_image is None:
        print(f"One or both images could not be loaded: {before_path}, {after_path}")
        return None

    h, w, _ = before_image.shape
    pose_landmarks = get_landmarks(before_image)

    if not pose_landmarks:
        raise ValueError("体のランドマークが検出されませんでした。")
    try:
        # Attempt to extract body coordinate
        body_coords = extract_body_region(before_image, pose_landmarks, h, w)
        body_coords = [(int(x), int(y)) for x, y in body_coords]  # 座標を整数化

        if body_coords is None:
            raise ValueError("Body coordinates could not be determined.")

    except Exception as e:
        # Handle exceptions and continue processing other images
        print(f"Error processing image pair {before_path} and {after_path}: {e}")

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

    # ディレクトリ内のファイルをリスト化（適切なエンコーディングでファイル名を取得）
    before_images = list_files_with_correct_encoding(before_dir)  # 文字化けに対応
    after_images = list_files_with_correct_encoding(after_dir)  # 文字化けに対応

    if len(before_images) != len(after_images):
        raise ValueError("beforeとafterの画像数が一致しません。")

    for before_img, after_img in zip(before_images, after_images):
        print("学習中… ")
        # before_dir と after_dir を適切に結合（絶対パスを使って）
        before_path = os.path.join(before_dir, before_img)  # ここで絶対パスのディレクトリとファイルを結合
        after_path = os.path.join(after_dir, after_img)     # 同様に

        try:
            print(f"Processing {before_img} and {after_img}...")
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

    return pd.DataFrame(data)

def train_sgd_regressor_model(df):
    """特徴量からRGB係数を予測するSGD回帰モデルを訓練します。"""
    X = df[['image_brightness', 'cheek_brightness', 'body_saturation', 'body_contrast']].values
    Y = df[['coeff_r', 'coeff_g', 'coeff_b']].values

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

    model_r = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42, alpha=0.01)
    model_g = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42, alpha=0.01)
    model_b = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42, alpha=0.01)

    model_r.fit(X_train, Y_train[:, 0])
    model_g.fit(X_train, Y_train[:, 1])
    model_b.fit(X_train, Y_train[:, 2])

    # テストデータでの予測
    Y_pred_r = model_r.predict(X_test)
    Y_pred_g = model_g.predict(X_test)
    Y_pred_b = model_b.predict(X_test)
    
    # 各ターゲット変数のMSEを計算
    mse_r = mean_squared_error(Y_test[:, 0], Y_pred_r)
    mse_g = mean_squared_error(Y_test[:, 1], Y_pred_g)
    mse_b = mean_squared_error(Y_test[:, 2], Y_pred_b)
    
    print(f"SGD回帰モデルのテストデータに対するMSE:")
    print(f"  coeff_r: {mse_r}")
    print(f"  coeff_g: {mse_g}")
    print(f"  coeff_b: {mse_b}")

    return {
        'model_r': model_r, 
        'model_g': model_g, 
        'model_b': model_b,
        'scaler_X': scaler_X,
        'scaler_Y': scaler_Y
    }

def train_and_save_model(sgd_model_save_path='trained_4factor_model3.pkl'):
    # 学習データのディレクトリ
    before_directory = 'training_data/before'
    after_directory = 'training_data/after'
    
    # 学習データの準備
    training_df = prepare_training_data(before_directory, after_directory)
    print(training_df.head())  # データフレームの内容を確認

    if training_df is None:
        print("new_data_df is None")
    else:
        print(training_df.head())
    
    # モデルを訓練（SGD回帰）
    sgd_model = train_sgd_regressor_model(training_df)
    
    # モデルをファイルに保存
    with open(sgd_model_save_path, 'wb') as file:
        pickle.dump(sgd_model, file)
    print(f"予測モデルを保存しました: {sgd_model_save_path}")

train_and_save_model()
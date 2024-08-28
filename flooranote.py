import os
import cv2
import json

# ディレクトリパスの指定
base_dir = 'corridor'
raw_image_dir = os.path.join(base_dir, 'raw_image')
ground_truth_dir = os.path.join(base_dir, 'ground_truth')
output_dir = os.path.join(base_dir, 'annotations')

# 出力ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

# すべてのraw_imageファイルに対して処理を行う
for image_filename in os.listdir(raw_image_dir):
    # 画像とマスクのパスを取得
    image_path = os.path.join(raw_image_dir, image_filename)
    mask_filename = image_filename.replace('.png', '_mask.png')  # マスクファイル名を推定
    mask_path = os.path.join(ground_truth_dir, mask_filename)
    
    # 画像とマスクが存在するか確認
    if not os.path.exists(mask_path):
        print(f"対応するマスクが見つかりません: {mask_filename}")
        continue
    
    # 画像とマスクを読み込む
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # マスクから輪郭を抽出し、アノテーションを作成
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotations = []
    
    for contour in contours:
        contour_points = contour.squeeze().tolist()
        annotations.append({
            'label': 'floor',
            'points': contour_points
        })

    # アノテーションデータの作成
    annotation_data = {
        'image_file': os.path.basename(image_path),
        'annotations': annotations
    }

    # 出力ファイルパス
    output_json_path = os.path.join(output_dir, image_filename.replace('.png', '.json'))

    # JSONファイルとして保存
    with open(output_json_path, 'w') as json_file:
        json.dump(annotation_data, json_file, indent=4)

    print(f"{output_json_path} を作成しました。")

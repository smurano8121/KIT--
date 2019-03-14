import numpy as np
import sys
import cv2
from flask import Flask, jsonify, abort, make_response

api = Flask(__name__)

# GETの実装
@api.route('/get', methods=['GET'])
def get():
    # 分類器
    cascade_path = "./lib/haarcascade_frontalface_default.xml"

    # 分類器の特徴量を取得
    cascade = cv2.CascadeClassifier(cascade_path)

    # Weカメラの設定
    cap = cv2.VideoCapture(0)

    if cap.isOpened() is False:
        print("can not open camera")
        sys.exit()

    elif cap.isOpened() is True:
        face_judge = 0 #0の時は顔が認識されていない
        ret, frame = cap.read()
        frame = cv2.resize(frame, (int(frame.shape[1]*0.7), int(frame.shape[0]*0.7)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        facerect = cascade.detectMultiScale(
            gray,
            scaleFactor=1.11,
            minNeighbors=3,
            minSize=(100, 100)
        )

        # 顔枠トリミング用の色指定
        color = (255,255,255)

        if len(facerect) != 0:
            face_judge = 1
        else:
            face_judge = 0
        print(face_judge)

    cap.release()
    result = { "FaceJudge_Result": face_judge }
    return make_response(jsonify(result))

# エラーハンドリング
@api.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

# ホスト0.0.0.0, ポート3001番でサーバーを起動
if __name__ == '__main__':
    api.run(host='0.0.0.0', port=3001)

#############
# 分類器
# cascade_path = "./lib/haarcascade_frontalface_default.xml"
#
# # 分類器の特徴量を取得
# cascade = cv2.CascadeClassifier(cascade_path)
#
# # Weカメラの設定
# cap = cv2.VideoCapture(0)
#
# if cap.isOpened() is False:
#     print("can not open camera")
#     sys.exit()
#
# while True:
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, (int(frame.shape[1]*0.7), int(frame.shape[0]*0.7)))
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     facerect = cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.11,
#         minNeighbors=3,
#         minSize=(100, 100)
#     )
#
#     # 顔枠トリミング用の色指定
#     color = (255,255,255)
#
#     if len(facerect) != 0:
#         for x, y, w, h in facerect:
#             # 顔検出した部分に枠を描画
#             cv2.rectangle(
#                 frame,
#                 (x, y),
#                 (x + w, y + h),
#                 color,
#                 thickness=2
#             )
#             face_judge = 1
#     else:
#         face_judge = 0
#
#     print(face_judge)
#     cv2.imshow('frame', frame)
#
#     # キー入力を1ms待って、k が27（ESC）だったらBreakする
#     k = cv2.waitKey(1)
#     if k == 27:
#         break
#
# # キャプチャをリリースして、ウィンドウをすべて閉じる
# cap.release()
# cv2.destroyAllWindows()
################

################
# 画像の顔認証

# 読み込むファイル → 出力先
# image_file = "person.jpg"
# output_path = "./outputs/"+ image_file

    # ファイルの読み込み＋グレースケール
    # img = cv2.imread(image_file)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 顔認識の実行
    # facerect = cascade.detectMultiScale(
    #         gray, scaleFactor=1.1,
    #         minNeighbors=2,
    #         minSize=(30, 30)
    # )

    # 顔枠トリミング用の色指定
    # color = (255,255,255)

    # if len(facerect) > 0:
    #
    #     #検出した顔を囲む矩形の作成
    #     for rect in facerect:
    #         cv2.rectangle(img, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
    #
    #     #認識結果の保存
    #     cv2.imwrite(output_path, img)
################

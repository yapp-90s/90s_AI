from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt
from gugongs.settings import AWS_STORAGE_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
import cv2
import numpy as np  # 행렬, 데이터 연산
from scipy.interpolate import UnivariateSpline
import boto3
import os
from imageFilter.filter import *


@csrf_exempt
@api_view(['POST'])
def apply(request):


    data = JSONParser().parse(request)
    film_code = data['film_code']
    film_uid = data['film_uid']
    photo_uid = data['photo_uid']

    #사진 저장 경로
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'imageFilter', 'images')  # current/imageFilter/images
    before_img = save_dir + '/%d.jpeg' % photo_uid
    after_img = save_dir + '/%d_edited.jpeg' % photo_uid



    # try:
    s3 = boto3.client('s3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key= AWS_SECRET_ACCESS_KEY)

    # 버킷 이름 / 다운로드 할 객체 지정 / 다운로드할 위치와 파일명
    s3.download_file(AWS_STORAGE_BUCKET_NAME,
                        '%d/%d.jpeg' % (film_uid, photo_uid),
                        before_img)
    
    print("5555")

    # 필터 적용
    image = cv2.imread(before_img, 1)  # 1 color, 2 grayscale, -1 alpha channel 포함

    if film_code == 1001:
        cv2.imwrite(after_img, gammaImage(image, 1.5))
    elif film_code == 1002:
        cv2.imwrite(after_img, warmImage(image))
    elif film_code == 1003:
        cv2.imwrite(after_img, coldImage(image))
    else:
        cv2.imwrite(after_img, guassian(image))
            
    

    

    # 업로드 할 파일 / 버킷 이름 / 업로드될 객체
    s3.copy_object(
        Bucket=AWS_STORAGE_BUCKET_NAME, 
        CopySource={'Bucket' : AWS_STORAGE_BUCKET_NAME, 'Key': '%d/%d.jpeg' % (film_uid, photo_uid)},
        Key='%d/%d_orgin.jpeg' % (film_uid, photo_uid))
    s3.delete_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key='%d/%d.jpeg' % (film_uid, photo_uid))
    
    s3.upload_file(after_img,AWS_STORAGE_BUCKET_NAME, '%d/%d.jpeg' % (film_uid, photo_uid))

    # 처리 끝난 이미지 프로젝트에서 삭제
    if os.path.exists(before_img):
        os.remove(before_img)
    if os.path.exists(after_img):
        os.remove(after_img)
    return Response(True)
    # except:
    #     return Response(False)



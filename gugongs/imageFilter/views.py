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



    try:
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
            cv2.imwrite(after_img, warmImage(image, 1.5))
        elif film_code == 1003:
            cv2.imwrite(after_img, coldImage(image, 1.5))
        else:
            cv2.imwrite(after_img, guassian(image, 1.5))
                
        

        

        # 업로드 할 파일 / 버킷 이름 / 업로드될 객체
        s3.upload_file(after_img,
                        AWS_STORAGE_BUCKET_NAME,
                        '%d/%d_edited.jpeg' % (film_uid, photo_uid))

        # 처리 끝난 이미지 프로젝트에서 삭제
        if os.path.exists(before_img):
            os.remove(before_img)
        if os.path.exists(after_img):
            os.remove(after_img)
        return Response(True)
    except:
        return Response(False)


def spreadLookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


def warmImage(img):
    redLT = spreadLookupTable([0, 65, 127, 256], [0, 100, 174, 256])
    blueLT = spreadLookupTable([0, 65, 127, 256], [0, 50, 100, 256])
    # 이미지 채널 분리
    blue, green, red = cv2.split(img)
    red = cv2.LUT(red, redLT).astype(np.uint8)
    blue = cv2.LUT(blue, blueLT).astype(np.uint8)
    # LUT테이블 적용 후 다시 병합
    return cv2.merge((blue, green, red))


def coldImage(img):
    blueLT = spreadLookupTable([0, 82, 138, 256], [0, 94, 152, 256])
    greenLT = spreadLookupTable([0, 65, 118, 256], [0, 78, 139, 256])
    blue, green, red = cv2.split(img)
    #    blue = cv2.LUT(blue, blueLT).astype(np.uint8)
    green = cv2.LUT(green, greenLT).astype(np.uint8)
    return cv2.merge((blue, green, red))


def gammaImage(img, gamma):
    img = img.astype(np.float)
    img = ((img / 255) ** (1 / gamma)) * 255
    gamma_img = img.astype(np.uint8)
    return gamma_img


def make_noise(std, img):  # std값이 클 수록 잡음이 크게 생성
    height, width = img.shape  # image의 width, height값 가짐
    noise_img = np.zeros((height, width), np.uint8)
    for i in range(height):
        for a in range(width):
            random_noise = np.random.normal()  # 랜덤함수를 이용하여 노이즈 적용
            set_noise = std * random_noise  # 랜덤값 * 잡음 곱함
            noise_img[i][a] = img[i][a] + set_noise  # imag_noise에 이미지의 원래 화소+set noise
    return noise_img


def blur(img):
    kernel = np.ones((11, 11), np.float32) / 121
    blur_img = cv2.filter2D(img, -1, kernel)
    return blur_img


def guassian(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이로 변환
    guassian_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    return guassian_img


def bilateral(img):
    bilateral_img = cv2.bilateralFilter(img, 9, 75, 75)
    return bilateral_img


def equalize(img):
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # 밝기 성분(Y)에 대해서만 히스토그램 평활화 수행
    ycrcb_planes = cv2.split(img_ycrcb)
    ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])
    dst_ycrcb = cv2.merge(ycrcb_planes)
    equal_image = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)
    return equal_image

import cv2
import numpy as np  # 행렬, 데이터 연산
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


def spreadLookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


"""
================================================================
LUT 이용
red_channel의 값을 상승
blue_channel의 값을 하향
전체적으로 따뜻한 이미지용
================================================================
"""


def warmImage(image):
    redLT = spreadLookupTable([0, 65, 127, 256], [0, 100, 174, 256])
    blueLT = spreadLookupTable([0, 65, 127, 256], [0, 50, 100, 256])
    # 이미지 채널 분리
    blue, green, red = cv2.split(image)
    red = cv2.LUT(red, redLT).astype(np.uint8)
    blue = cv2.LUT(blue, blueLT).astype(np.uint8)
    # LUT테이블 적용 후 다시 병
    return cv2.merge((blue, green, red))


"""
================================================================
LUT 이용
red_channel의 값을 하향
blue_channel의 값을 상향
전체적으로 차가운 이미지
================================================================
"""


def coldImage(image):
    blueLT = spreadLookupTable([0, 82, 138, 256], [0, 94, 152, 256])
    greenLT = spreadLookupTable([0, 65, 118, 256], [0, 78, 139, 256])
    blue, green, red = cv2.split(image)
#    blue = cv2.LUT(blue, blueLT).astype(np.uint8)
    green = cv2.LUT(green, greenLT).astype(np.uint8)
    return cv2.merge((blue, green, red))


"""
================================================================
감마보정
output = input^gamma (감마값 1기준 높으면 어두워지고 낮으면 밝아짐)
오버플로우 방지 위해 0~1 값으로 바꾸는 정규화 필요
소수가 될 수 있게 감마값을 float
(out/255)로 모든 값 0~1사이로 정규화
**을 통해 (1/g)만큼 제곱 => 값이 높아질수록 밝기가 같이 올라감
================================================================
"""


def gammaImage(img, gamma):
    img = img.astype(np.float)
    img = ((img / 255) ** (1 / gamma)) * 255
    img = img.astype(np.uint8)
    return img


image = cv2.imread("/Users/hongjeongmin/OpenCV/image_filter/view_sample.jpeg", 1)  # 1 color, 2 grayscale, -1 alpha channel 포함
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #그레이로 변환
cv2.imshow('sea', image)
cv2.imshow('gray_sea', gray_image)
cv2.imshow('cold_sea', coldImage(image))
cv2.imshow('warm_sea', warmImage(image))
cv2.imshow('gamma_sea', gammaImage(image, 1.5))

"""
================================================================
커널을 사용한 이미지 필터링
1. 해당 픽셀 중심으로 5*5영역 만들기(영역 크기는 지정할 수 있음)
2. 커널과 해당 영역을 합쳐서 이 영역에 속하는 픽셀값과 커널을 곱해 그 값을 합친다
3. 더한값을 5*5인 25로 나누고 해당 픽셀값으로 초기화 한다
=> 커널이 커지면 커질수록 이미지가 흐릿해진다
================================================================
"""
kernel1 = np.ones((5, 5), np.float32) / 25  # average filter는 모든 행렬이 1인 행렬, ones는 1로 초기화
kernel2 = np.ones((11, 11), np.float32) / 121
blur1 = cv2.filter2D(image, -1, kernel1)  # depth -1은 원본이미지와 동일 값
blur2 = cv2.filter2D(image, -1, kernel2)
# cv2.imshow('blur1', blur1)
cv2.imshow('blur2', blur2)


"""
================================================================
Gaussian Filtering
가우시안 함수를 이용한 커널을 적용함
전체적으로 밀도가 높은 노이즈, 백색 노이즈 제거에 효과적
중앙에 가중치가 높음, 커널 사이즈는 무조건 홀수인 양수
================================================================
"""
img_gaussian = gray_image.copy()
guassian_blur = cv2.GaussianBlur(img_gaussian, (5, 5), 0)
cv2.imshow('guassian', guassian_blur)


"""
================================================================
bilateral Filtering
Bilateral filter도 Gaussian filter처럼 가중치를 적용한 MASK를 사용하지만
가중치에 중심 화소에서의 거리뿐만 아니라 중심 화소와의 밝기 차이도 고려한다

cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
Parameters:	
src – 8-bit, 1 or 3 Channel image
d – filtering 시 고려할 주변 pixel 지름
sigmaColor – Color를 고려할 공간. 숫자가 크면 멀리 있는 색도 고려함.
sigmaSpace – 숫자가 크면 멀리 있는 pixel도 고려함.
================================================================
"""

bilateral_img = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imshow('bilateral', bilateral_img)

"""
================================================================
Gaussian noise
================================================================
"""


def make_noise(std, img):  # std값이 클 수록 잡음이 크게 생성
    height, width = img.shape  # image의 width, height값 가짐
    img_noise = np.zeros((height, width), np.uint8)
    for i in range(height):
        for a in range(width):
            random_noise = np.random.normal()  # 랜덤함수를 이용하여 노이즈 적용
            set_noise = std * random_noise  # 랜덤값 * 잡음 곱함
            img_noise[i][a] = img[i][a] + set_noise  # imag_noise에 이미지의 원래 화소+set noise
    return img_noise



cv2.imshow('noise_image', make_noise(5, gray_image))


"""
================================================================
histogram Stretching
이미지를 보기 좋게 조정 => 특정 픽셀 밝기값이 집중되어있는 것을 퍼뜨려 가시도를 높임
픽셀의 최소, 최대 비율을 이용해 고정된 비율로 낮은 밝기와 높은 밝기로 당겨줌    
================================================================
"""
# 히스토 그램 계산: BGR채널 [1],[2],[3] /None: 전체 이미지에 대해/ 전체영역 계산 [256] / 범위 [0,256]
hist1 = cv2.calcHist([image], [0], None, [256], [0,256])
plt.plot(hist1)

norm_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
# cv2.imshow('normalize', norm_img) #스트레칭 후

"""
================================================================
histogram Equallization
이미지의 밝기가 알맞지 않은 경우 히스토그램을 전체 영역에 고루 퍼져 있도록 바꾸어주면 이미지가 개선
================================================================
"""
img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# 밝기 성분(Y)에 대해서만 히스토그램 평활화 수행
ycrcb_planes = cv2.split(img_ycrcb)
ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])
dst_ycrcb = cv2.merge(ycrcb_planes)
equal_image = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)
cv2.imshow('equalization', equal_image)


# hist2 = cv2.calcHist([equal_image], [0], None, [256], [0,256])
# plt.plot(hist2)
# plt.show()

cv2.waitKey(0)  # 키입력 대기
cv2.destroyAllWindows()

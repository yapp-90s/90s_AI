# 90s_AI

정민이가 만든 멋진 AI를 이용한 사진 후처리 API 입니다.

### URL
http://49.50.162.246:8082/imageFilter/apply/


### Request
Film_code 의 경우 1001, 1002, 1003, 1004 만 현재 가능.
````
{
    "film_code":Int,
    "film_uid": Int,
    "photo_uid": Int
}
````

### Response 
````
성공 : true
실패 : false
````

import json
import requests
_='''
# FastAPI
# root
root_url = 'http://127.0.0.1:8000/'
response_root = requests.get(root_url)
print(response_root.json())

# predict
predict_url = 'http://127.0.0.1:8000/predict'
data = {"text": "정말 재밌게 잘봤습니다!"}

predict_response = requests.post(predict_url, data=json.dumps(data))
print(predict_response.json())


# Flask
# root
root_url = 'http://127.0.0.1:5000/'
response_root = requests.get(root_url)
print(response_root.json())
'''
# predict
predict_url = 'http://127.0.0.1:5000/predict'
data = {"text": ["정말 재밌게 잘봤습니다!","뭐야뭐야뭐야뭐야 뭐야 뭐야","최고에요 정말 인생영화에요",
"크리스토퍼 감독님 사랑해요","천사 묻었네 왜 보냐","개창렬하다", "장원영 최고 여신님 사랑해요",
"미친구라즐"]}

header = {'Content-type':'application/json'}

predict_response = requests.post(predict_url, data=json.dumps(data), headers=header)
print(predict_response.json())


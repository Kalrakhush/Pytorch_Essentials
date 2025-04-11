import requests
resp = requests.post('http://localhost:5000/predict', files={'file':open('sample_image-300x298.png', 'rb')})

print(resp.text)

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Time':2, 'Amount':9})

print(r.json())
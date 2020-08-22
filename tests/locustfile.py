import time
import io
from locust import HttpUser, task, between

image_file = io.open('../images/locust.jpg', 'rb')
image = image_file.read()


class QuickstartUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def detect_expiration(self):
        files = {'image': image}
        response = self.client.post("/expiry-detection", files=files, verify=False)
        assert response.status_code == 200
        time.sleep(1)

    @task
    def detect_object(self):
        files = {'image': image}
        response = self.client.post("/object-detection", files=files, verify=False)
        assert response.status_code == 200
        time.sleep(1)

    @task
    def detect(self):
        files = {'image': image}
        response = self.client.post("/detection", files=files, verify=False)
        assert response.status_code == 200
        time.sleep(1)

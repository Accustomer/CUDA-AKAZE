# CUDA-AKAZE
AKAZE keypoints detection, descriptors computation and BF-matching by CUDA.

# Requirement
OpenCV, CUDA

# Result
Keypoints example:
![Keypoint](https://github.com/Accustomer/CUDA-AKAZE/blob/main/data/akaze_show1.jpg)

Matching example:
![Match](https://github.com/Accustomer/CUDA-AKAZE/blob/main/data/akaze_show_matched.jpg)

# Speed
Repeat the matching test for 100 times took an average of 19.56ms on NVIDIA GeForce GTX 1080:
![timecost-of-all](https://github.com/Accustomer/CUDA-AKAZE/blob/main/data/timecost.png)

Compare to [nbergst](https://github.com/nbergst/akaze), there's an improvement of 3.5ms:
![timecost](https://user-images.githubusercontent.com/46698134/224546520-02d06e03-fb1e-4dbc-aa70-508ff1dd2501.png)

# Reference
https://github.com/pablofdezalc/akaze

https://github.com/h2suzuki/fast_akaze

GitHub - nbergst/akaze: Accelerated-KAZE Features with CUDA acceleration



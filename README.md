In the process of setting up the environment for this project, I chose to use a Raspberry Pi 3 Model B Plus Rev 1.3 as the hardware platform, running the default Raspberry Pi OS based on Debian. Python 3.7.12 was selected as the programming language, and TensorFlow 2.4.0 was installed as the deep learning framework. However, I encountered challenges in ensuring compatibility between Python, TensorFlow, and the Raspberry Pi model. Specifically, I faced issues when trying to obtain the correct versions for TensorFlow and Python. Despite my efforts to install the appropriate versions and confirming compatibility with my 32-bit OS, I persistently encountered the error "No module named 'tensorflow'." After some troubleshooting, I was able to resolve this issue by opting to reinstall the entire environment. This corrective step has proven effective, and I was able to continue with the project. Additionally, the camera used for image capture in this project is the Pi Camera with the imx219 sensor, providing the necessary imaging capabilities despite the initial setup issues I encountered.I also had to make sure my 1.19.5 numpy version was compatible with my tensorflow.

Hardware: Raspberry Pi 3 Model B Plus Rev 1.3
Operating System: Raspberry Pi Debian - Default Pi OS 32 bit
Python: Python 3.7.12
TensorFlow: 2.4.0
Camera: Pi Cam-imx219

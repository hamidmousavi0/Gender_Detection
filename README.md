# Gender_Detection web service
0-install python on UBUNTU
	-Ubuntu 18.04 ships with Python 3, as the default Python installation
	-sudo apt install python3-pip
	-sudo apt-get install virtualenv
1- create virtual enviornment of python3.6
	-virtualenv .venv
	-source .venv/bin/activate
2-install packages
	-pip3 install tensorflow
	-pip3 install keras
	-pip3 install watchdog
	-git clone https://github.com/simplejson/simplejson.git
	-python3 setup.py install
	-pip3 install requests
	-pip3 insatll opencv-python
	-pip3 install matplotlib
	-pip3 install blinker
3-run server 
	-python3 server.py
4-client request
	-python3 client.py


**** in server.py should determine host and port and also set in client.py

# in server.py
	-app.run(host="0.0.0.0", port=5000)
# in clinet.py
	-addr = 'http://0.0.0.0:5000'

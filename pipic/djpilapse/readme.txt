Django Server für Fernzugriff starten:
direkt unter pycharm : runserver 192.168.2.118:8000
---------------------------------------------------------------------------------------------------------
Ansprechbare Seiten:

http://127.0.0.1:8000/admin/
http://127.0.0.1:8000/admin/login/?next=/admin/
http://127.0.0.1:8000/admin/djpilapp/pilapse_project/
http://192.168.2.118:8000/djpilapp/startlapse/
---------------------------------------------------------------------------------------------------------
Wie man ein Django Project von 'aussen her' über das LAN zugänglich macht

http://raspberrypituts.com/raspberry-pi-django-tutorial-2017/

im Setting file in der Zeile ALLOWED_HOSTS = [ ] die IP des Raspberry Pi's eintragen.

ALLOWED_HOSTS = ['localhost', '127.0.0.1', '192.168.2.118','www.mysite.com']

Anschliessend in der Virt Env auf dem Raspi den Server starten:

python manage.py runserver 0.0.0.0:8080
----------------------------------------------------------------------------------------------------------
Installed packages:
-------------------
pip install django-celery

pip install -U django

pip install -U celery

------------------------------------------------------------------------------------------------------------------------

To get the Django app running, try adding the following three lines to your Pi's crontab:

@reboot pi /usr/bin/screen -dmS tlapse python /home/pi/python_scripts/pipic/djpilapse/manage.py runserver 192.168.2.118:8000

@reboot pi /usr/bin/screen -dmS celery bash -c 'sleep 10; (cd /home/pi/python_scripts/pipic/djpilapse && exec celery -A djpilapse worker -l info )'

@reboot pi bash -c 'sleep 40; wget 192.168.2.118:8000/djpilapp/startlapse/'

You will also need to manually set your Pi's IP address to 192.168.0.5 for ethernet.
(You can actually use any value you like; just make sure the crontab lin have amatching IP address.) Then reboot.

You can then access the web interface in one of two ways. a) Open a browser on the Pi and go to 192.168.0.5:8000, or
b) Set your laptop to manual IP address 192.168.0.10 (or any 192.168.0.X with X not equal to 5), connect an ethernet
cable to the Pi, and then open a browser and navigate to 192.168.0.5:8000.

-----------------------------------------------------------------------------------------------------------------------
Screen : ist ein Fenstermanager zur Verwendung mit textbasierten Eingabefenstern (Textkonsole). Hierbei ist es möglich,
innerhalb eines einzigen Zugangs (zum Beispiel über ein Terminal oder einer Terminalemulation) mehrere virtuelle
Konsolensitzungen zu erzeugen und zu verwalten. Darüber hinaus können Sitzungen getrennt und später fortgeführt werden.

[ -d -m   Start screen in "detached" mode. This creates a new session but doesn't attach to it. This is useful for system startup scripts. ]

wget: command line tools that can download contents from FTP, HTTP and HTTPS

tlapse: A tiny utility that takes periodic screenshots of your site while you develop.

-----------------------------------------------------------------------------------------------------------------------
Celery - Error:

Wenn python /home/pi/python_scripts/pipic/djpilapse/manage.py runserver 192.168.2.118:8000 dann folgt der Fehler:

[2018-01-16 13:39:20,023: ERROR/MainProcess] consumer: Cannot connect to amqp://guest:**@127.0.0.1:5672//: [Errno 111] Connection refused.
Trying again in 28.00 seconds...

http://open-edx.readthedocs.io/en/latest/amazon_snapshot_bug.html

Istallation von rabbitmq-server :
nach : http://blog.abarbanell.de/raspberry/2015/06/06/making-of-benchmarking-rabbitmq-on-raspberry/

Configuraton Rabbitmq - Server:
sudo rabbitmqctl add_user pi 123ihomelab
sudo rabbitmqctl set_user_tags pi administrator
sudo rabbitmqctl set_permissions -p / pi ".*" ".*" ".*"
sudo rabbitmq-plugins enable rabbitmq_management

Falls alles korrekt installiert wurde dann kann man sich auf der Mnagement-Site via Browser einloggen: http://192.168.2.118:15672/#/  user: pi | pw: 123ihomelab

Für eine ausführliche Erklärung siehe in den Evernotes unter : Raspbian Stretch -> Django & Celery
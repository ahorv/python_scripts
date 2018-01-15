Ansprechbare Seiten:

http://127.0.0.1:8000/admin/
http://127.0.0.1:8000/admin/djpilapp/pilapse_project/
---------------------------------------------------------------------------------------------------------
Wie man ein Django Project von 'aussen her' über das LAN zugänglich macht

http://raspberrypituts.com/raspberry-pi-django-tutorial-2017/

im Setting file in der Zeile ALLOWED_HOSTS = [ ] die IP des Raspberry Pi's eintragen.

ALLOWED_HOSTS = ["localhost","10.0.0.22"]

Anschliessend in der Virt Env auf dem Raspi den Server starten:

python manage.py runserver 0.0.0.0:8080
----------------------------------------------------------------------------------------------------------
There are a couple pre-requisites, though. Use the python 'pip' installer to make sure you have recent versions
of each.

sudo apt-get install python-pip

Then:

pip install -U django

pip install -U celery

To get the Django app running, try adding the following three lines to your Pi's crontab:

@reboot pi /usr/bin/screen -dmS tlapse python /home/pi/python_scripts/pipic/djpilapse/manage.py runserver 192.168.0.5:8000

@reboot pi /usr/bin/screen -dmS celery bash -c 'sleep 10; (cd /home/pi/python_scripts/pipic/djpilapse && exec celery -A djpilapse worker -l info )'

@reboot pi bash -c 'sleep 40; wget 192.168.2.118:8000/djpilapp/startlapse/'

You will also need to manually set your Pi's IP address to 192.168.0.5 for ethernet.
(You can actually use any value you like; just make sure the crontab lin have amatching IP address.) Then reboot.

You can then access the web interface in one of two ways. a) Open a browser on the Pi and go to 192.168.0.5:8000, or
b) Set your laptop to manual IP address 192.168.0.10 (or any 192.168.0.X with X not equal to 5), connect an ethernet
cable to the Pi, and then open a browser and navigate to 192.168.0.5:8000.
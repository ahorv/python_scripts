#!/usr/bin/env python3s
import os
import sys
import logging
import logging.handlers
import ftplib
import shutil
from glob import glob

#########################################################################
##  02.11.2017 Version 1 : rawexporter.py
#########################################################################
# Exports rawpictures (zipfolder with raw pictures) to the ihomelab ftp.
# All .*zip send to ftp-server will be deleted after sending !
# ftp addr: ftp.ihomelab.ch / resolved: 147.88.219.198:21
# Path to directory on the ftp server: /camera_2/raw/
#
# NEW:
# -----
#
#########################################################################


# global variables
global FTPADR
global SCRIPTPATH
global ERRFILEPATH
global ZIPDIRPATH


if sys.platform == "linux":
    import pwd
    import grp

    SCRIPTPATH  = os.path.join('/home', 'pi', 'python_scripts', 'helpers')
    ERRFILEPATH = os.path.join(SCRIPTPATH, 'rawexporter.log')
    ZIPDIRPATH = os.path.join('/home', 'pi', 'python_scripts', 'raw', 'raw_pictures')
else:
    SCRIPTPATH  = os.path.realpath(__file__)
    ERRFILEPATH = os.path.join(SCRIPTPATH, 'rawexporter.log')
    ZIPDIRPATH  = os.path.join(SCRIPTPATH, 'raw_pictures')


class Logger:
    def getLogger(self):

        try:
            global ERRFILEPATH

            # configure log formatter
            logFormatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

            # configure file handler
            fileHandler = logging.FileHandler(ERRFILEPATH)
            fileHandler.setFormatter(logFormatter)

            # configure stream handler
            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(logFormatter)

            # get the logger instance
            self.logger = logging.getLogger(__name__)

            # set rotating filehandler
            handler = logging.handlers.RotatingFileHandler(ERRFILEPATH, encoding='utf8',
                                                           maxBytes=1024 * 10000, backupCount=1)

            # set the logging level
            self.logger.setLevel(logging.INFO)

            if not len(self.logger.handlers):
                self.logger.addHandler(fileHandler)
                self.logger.addHandler(consoleHandler)
                # self.logger.addHandler(handler)

            setOwnerAndPermission(ERRFILEPATH)
            return self.logger

        except IOError as e:
            print('Error logger:' + str(e))

def setOwnerAndPermission(pathToFile):
    try:

        if sys.platform == "linux":
            uid = pwd.getpwnam('pi').pw_uid
            gid = grp.getgrnam('pi').gr_gid
            os.chown(pathToFile, uid, gid)
            os.chmod(pathToFile, 0o777)
        else:
            return
    except IOError as e:
        print('PERM : Could not set permissions for file: ' + str(e))


class Exporter:
    def grabAllRawZip(self):
        global ZIPDIRPATH
        try:
            s = Logger()
            root_logger = s.getLogger()

            allzipfiles = []

            for file in sorted(glob(ZIPDIRPATH +'/*.zip')):
                allzipfiles.append(file)

            return allzipfiles

        except Exception as e:
            root_logger.error('GRAB: Error: ' + str(e))

    def sendZipToFTP(self):
        global db_con
        global db_path

        success = False

        try:
            s = Logger()
            root_logger = s.getLogger()

            e = Exporter()
            allzipfiles = e.grabAllRawZip()
            uploadedzip = []
            cnt = 0

            for zip_file in sorted(allzipfiles):

                zipfilename = zip_file.split('/')[-1]
                cnt = cnt + 1

                ftp = ftplib.FTP('ftp.ihomelab.ch')
                ftp.login('tahorvat', '123ihomelab')
                ftp.cwd('/camera_2/raw/')

                zipfile = open(zip_file, 'rb')  # file to send
                ftp.storbinary('STOR %s' % zipfilename, zipfile)  # send the file
                zipfile.close()  # close file and FTP

                uploadedzip.append(zip_file)
                success = True

                ftp.quit()

            root_logger.info(' FTP : {} *.zip files uploaded to ftp.ihomelab.ch'.format(cnt))
            e.deleteUploadedZip(uploadedzip)
            return success

        except Exception as e:
            root_logger.error('FTP : Error: ' + str(e))
            return False

    def deleteUploadedZip(self,uploadedZipFiles):
        try:
            s = Logger()
            root_logger = s.getLogger()
            cnt = 0

            for zip_file in uploadedZipFiles:
                cnt = cnt + 1
                os.remove(zip_file)
                success = True

            root_logger.info('DEL : {} *.zip removed '.format(cnt))

            return success

        except Exception as e:
            root_logger.error('DEL : Error: ' + str(e))
            return False

def main():
    try:
        s = Logger()
        root_logger = s.getLogger()
        root_logger.info('Main: UPLOAD STARTED')
        e = Exporter()
        e.sendZipToFTP()

    except Exception as e:
        root_logger.error('MAIN: Error: ' + str(e))


if __name__ == '__main__':
    main()
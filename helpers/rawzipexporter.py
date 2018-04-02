#!/usr/bin/env python
import os
import sys
import time
import zipfile
import logging
import logging.handlers
import shutil
from ftplib import FTP, error_perm
from glob import glob

#########################################################################
##  26.03.2018 Version 1 : rawzipexporter.py
#########################################################################
# New Version of the former rawexporter.py
#
# Collects all image directories, zips and sends them to the ihomelab ftp.
# All directories will be deleted after zipping.
# ftp addr: ftp.ihomelab.ch / resolved: 147.88.219.198:21
# Path to directory on the ftp server: /camera_2/raw/
#
# The variable CAMERA has to be set according to the camera in use!
#
# NEW:
# -----
# - new directory for each day
# - added variable to distinguish between cameras (must be set accordingly !)
# - added zipping
#
#########################################################################


# global variables
global FTPADR
global SCRIPTPATH
global ERRFILEPATH
global ZIPDIRPATH
global CAMERA

CAMERA = 'camera_1'

if sys.platform == "linux":
    import pwd
    import grp

    SCRIPTPATH = os.path.join('/home', 'pi', 'python_scripts', 'helpers')
    ERRFILEPATH = os.path.join(SCRIPTPATH, 'rawexporter.log')
    ZIPDIRPATH = os.path.join('/home', 'pi', 'python_scripts', 'raw', 'raw_data')
else:
    SCRIPTPATH = os.path.realpath(__file__)
    ERRFILEPATH = os.path.join(SCRIPTPATH, 'rawzipexporter.log')
    ZIPDIRPATH = os.path.join(SCRIPTPATH, 'raw_data')


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

    def getDirs(self):
        try:
            global ZIPDIRPATH
            s = Logger()
            root_logger = s.getLogger()
            allDirs = []

            for dirs in sorted(glob(os.path.join(ZIPDIRPATH, "*", ""))):
                if os.path.isdir(dirs):
                    allDirs.append(dirs)

            return allDirs

        except Exception as e:
            root_logger.info('GETDIRS: Error: ' + str(e))

    def zipitall(self):
        try:
            global ZIPDIRPATH
            s = Logger()
            root_logger = s.getLogger()

            allDirs = []
            allDirs = self.getDirs()

            for dirs in allDirs:

                for nextdir, subdirs, files in os.walk(dirs):
                    newzipname = nextdir.split('/')[-2]

                    if newzipname:

                        zipfilepath = os.path.join(ZIPDIRPATH, newzipname + '.zip')

                        zf = zipfile.ZipFile(zipfilepath, "w")

                        for dirname, subdirs, files in os.walk(dirs):
                            for filename in files:
                                zf.write(os.path.join(dirname, filename), filename, compress_type=zipfile.ZIP_DEFLATED)

                        zf.close()

                        # delete zipped directory
                        shutil.rmtree(dirs, ignore_errors=True)

        except IOError as e:
            root_logger.error('ZIPALL :Error: ' + str(e))


    def grabAllRawZip(self):
        global ZIPDIRPATH
        try:
            s = Logger()
            root_logger = s.getLogger()

            allzipfiles = []

            for file in sorted(glob(ZIPDIRPATH + '/*.zip')):
                allzipfiles.append(file)

            return allzipfiles

        except Exception as e:
            root_logger.error('GRAB: Error: ' + str(e))

    def sendZipToFTP(self):

        success = False

        try:
            uploadedzip = []
            cnt = 0

            s = Logger()
            root_logger = s.getLogger()

            time_to_zip_start = time.time()
            self.zipitall()
            time_to_zip_end = time.time()
            time_to_zip = time_to_zip_end-time_to_zip_start
            root_logger.info('Time to zip all files: {}'.format(round(time_to_zip,1)))

            allzipfiles = self.grabAllRawZip()
            zipfilename = allzipfiles[0]
            token = str(zipfilename.split('/')[-1])
            newDirName = str(token.split('_', 1)[0])

            ftpPath = '/' + CAMERA + '/raw/'

            ftp = FTP('ftp.ihomelab.ch')
            ftp.login('tahorvat', '123ihomelab')
            ftp.cwd(ftpPath)

            try:
                ftp.cwd(newDirName)

            except error_perm:
                ftp.mkd(newDirName)
                ftp.cwd(newDirName)

            time_to_send_start = time.time()
            for zip_file in sorted(allzipfiles):
                zipfilename = zip_file.split('/')[-1]
                cnt = cnt + 1

                with open(zip_file, 'rb') as out:
                    ftp.storbinary('STOR ' + zipfilename, out)

                uploadedzip.append(zip_file)
                success = True

            ftp.close()
            time_to_send_end = time.time()
            time_to_send = time_to_send_end-time_to_send_start

            root_logger.info('FTP : {} zip files uploaded in {} sec'.format(cnt, round(time_to_send,1)))
            self.deleteUploadedZip(uploadedzip)
            return success

        except Exception as e:
            root_logger.error('FTP : Error: ' + str(e))
            ftp.quit()
            return False

    def deleteUploadedZip(self, uploadedZipFiles):
        try:
            s = Logger()
            root_logger = s.getLogger()
            cnt = 0

            for zip_file in uploadedZipFiles:
                cnt = cnt + 1
                os.remove(zip_file)
                success = True

            root_logger.info('DEL : {} zip removed '.format(cnt))

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

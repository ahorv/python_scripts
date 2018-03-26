#!/usr/bin/env python
import os
import sys
import zipfile
import logging
import logging.handlers
import shutil
from ftplib import FTP, error_perm
from glob import glob

#########################################################################
##  26.03.2018 Version 3 : rawzipexporter.py
#########################################################################
# New Version of the former rawexporter.py
#
# Collects all image directories, zips and sends them to the ihomelab ftp.
# All directories will be deleted after zipping.
# ftp addr: ftp.ihomelab.ch / resolved: 147.88.219.198:21
# Path to directory on the ftp server: /camera_2/raw/
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

CAMERA = 'camera_3'

'''
if sys.platform == "linux":
    import pwd
    import grp

    SCRIPTPATH = os.path.join('/home', 'pi', 'python_scripts', 'helpers')
    ERRFILEPATH = os.path.join(SCRIPTPATH, 'rawexporter.log')
    ZIPDIRPATH = os.path.join('/home', 'pi', 'python_scripts', 'raw', 'raw_data')
else:
    SCRIPTPATH = os.path.realpath(__file__)
    ERRFILEPATH = os.path.join(SCRIPTPATH, 'rawexporter.log')
    ZIPDIRPATH = os.path.join(SCRIPTPATH, 'raw_data')
'''

SCRIPTPATH = r'C:\Users\ati\Desktop\raw_data'
ERRFILEPATH = r'C:\Users\ati\Desktop\raw_data\rawexporter.log'
ZIPDIRPATH = SCRIPTPATH

# ACHTUNG setPermission ist AUSKOMMENTIERT !
# ACHTUNG FTP UPLOAD Auskommentiert !

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
        return

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
            allDirs = []

            for dirs in sorted(glob(os.path.join(ZIPDIRPATH, "*", ""))):
                if os.path.isdir(dirs):
                    allDirs.append(dirs)
                    print(dirs)

            return allDirs

        except Exception as e:
            print('getDirectories: Error: ' + str(e))

    def zipitall(self):
        try:
            s = Logger()
            root_logger = s.getLogger()

            allDirs = []
            allDirs = self.getDirs()

            for dirs in allDirs:

                print('Zipping directory: {}'.format(dirs))

                dirtozip = ''
                for nextdir, subdirs, files in os.walk(dirs + "/"):
                    newzipname = nextdir.split('/')[-1]
                    if newzipname:

                        dirtozip = os.path.join(dirs, newzipname)
                        zipfilepath = os.path.join(dirs, newzipname)

                        zf = zipfile.ZipFile(zipfilepath + '.zip', "w")
                        for dirname, subdirs, files in os.walk(dirtozip):
                            print('{}:'.format(dirname))
                            for filename in files:
                                print('\t'+'- {}'.format(filename))
                                zf.write(os.path.join(dirname, filename), filename, compress_type=zipfile.ZIP_DEFLATED)

                        zf.close()

                        # delete zipped directories
                        shutil.rmtree(dirtozip, ignore_errors=True)

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
            s = Logger()
            root_logger = s.getLogger()

            e = Exporter()

            e.zipitall()

            allzipfiles = e.grabAllRawZip()
            uploadedzip = []
            cnt = 0

            zipfilename = allzipfiles[0]
            token = str(zipfilename.split('/')[-1])
            newDirName = str(token.split('_', 1)[0])
            print('New Dir Name: ' + newDirName)

            #############################################
            #   ACHTUNG FTP AUSKOMMENTIERT
            #############################################
            return

            '''
            ftpPath = '/' + CAMERA + '/raw/'

            ftp = FTP('ftp.ihomelab.ch')
            ftp.login('tahorvat', '123ihomelab')
            ftp.cwd(ftpPath)
            '''

            try:
                ftp.cwd(newDirName)

            except error_perm:
                ftp.mkd(newDirName)
                ftp.cwd(newDirName)

            for zip_file in sorted(allzipfiles):
                zipfilename = zip_file.split('/')[-1]
                cnt = cnt + 1

                with open(zip_file, 'rb') as out:
                    # print('FTP STOR: %s' % ftp.storbinary('STOR ' + zipfilename, out))
                    ftp.storbinary('STOR ' + zipfilename, out)

                uploadedzip.append(zip_file)
                success = True

            ftp.close()

            root_logger.info(' FTP : {} *.zip files uploaded to ftp.ihomelab.ch'.format(cnt))
            e.deleteUploadedZip(uploadedzip)
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

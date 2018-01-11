import os
import random
import argparse

class ScanFile(object):
    def __init__(self,directory,prefix=None,postfix='.jpg'):
        self.directory=directory
        self.prefix=prefix
        self.postfix=postfix

    def scan_files(self):
        files_list=[]

        for dirpath,dirnames,filenames in os.walk(self.directory):
            '''''
            dirpath is a string, the path to the directory.
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..').
            filenames is a list of the names of the non-directory files in dirpath.
            '''
            for special_file in filenames:
                if self.postfix:
                    special_file.endswith(self.postfix)
                    files_list.append(os.path.join(dirpath,special_file))
                elif self.prefix:
                    special_file.startswith(self.prefix)
                    files_list.append(os.path.join(dirpath,special_file))
                else:
                    files_list.append(os.path.join(dirpath,special_file))

        return files_list

    def scan_subdir(self):
        subdir_list=[]
        for dirpath,dirnames,files in os.walk(self.directory):
            subdir_list.append(dirpath)
        return subdir_list

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Parser added')
    parser.add_argument('-s',
        action="store",
        dest="dir_src",
        default="/home/user/Documents/yida/database/annotated_data/annotated_img", 
        help='The path for input images')
    parser.add_argument('-t',
        action="store",
        dest="dir_tar",
        default="/home/user/Documents/yida/database/annotated_data/annotated_obj", 
        help='The path for target images')
    parser.add_argument('-o',
        action="store",
        dest="output_file",
        default="./list_annotated_shapenet.csv", 
        help='Output file in csv format')
    parser.print_help()
    results = parser.parse_args()
    
    dir_src=results.dir_src
    dir_tar=results.dir_tar
    scan=ScanFile(dir_src)
    subdirs=scan.scan_subdir()
    files=scan.scan_files()

    print("The subdirs scaned are:")
    for subdir in subdirs:
        print(subdir)

    file_object = open(results.output_file, 'w')
    # shuffle the list
    random.seed(1)
    random.shuffle(files)

    for file in files:
        file_tar = file.replace(dir_src, dir_tar, 1)
        label_start = file.rfind('/')+1
        label_end = file.find('_', label_start)
        file_object.write(file + ',' + file_tar + ','
                + str(int(file[label_start:label_end])) + '\n')
    print("The files scaned are already written in csv")

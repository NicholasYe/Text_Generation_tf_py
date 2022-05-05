import re
import os
import sys

path_in = 'bailuyuan.txt'
path_out = 'test.txt'

def updateFile(file,old_str,new_str):
    """
    替换文件中的字符串
    :param file:文件名
    :param old_str:旧字符串
    :param new_str:新字符串
    :return:
    """
    file_data = ""
    with open(file,"r",encoding='utf-8') as f:
        for line in f:
            line = line.replace(old_str,new_str)
            file_data += line
    with open(file,"w",encoding='utf-8') as f:
        f.write(file_data)

def delblankline(infile, outfile):
    infopen = open(infile,'r',encoding="utf-8")
    outfopen = open(outfile,'w',encoding="utf-8")
    db = infopen.read().lower()
    outfopen.write(db.replace('。','。\n'))
    infopen.close()
    outfopen.close()

delblankline(path_in,path_out)

import os
import sys

#sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath('__file__') )))

cur_dir = os.path.join(*os.getcwd().replace(':','').split('\\')[:-1])
sys.path.insert(0, cur_dir[0]+':'+cur_dir[1:])
sys.path.append('..')

from Contrarian-Trading-Strategies.utils import getData
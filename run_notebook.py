"""Execute the notebook in a fresh Python kernel, preserving real outputs."""
from pathlib import Path
import json, os, sys, argparse
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parent
kernel=root/'.local_jupyter/kernels/day4-project'
kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
    'display_name':'Day 4 project','language':'python'}))
os.environ['JUPYTER_PATH']=str(root/'.local_jupyter')
os.environ['JUPYTER_RUNTIME_DIR']=str(root/'.local_jupyter/runtime')
os.environ['MPLCONFIGDIR']=str(root/'.local_matplotlib')
os.environ['IPYTHONDIR']=str(root/'.local_ipython')
path=root/'notebook/neural_network.ipynb'
nb=nbformat.read(path,as_version=4)
parser=argparse.ArgumentParser()
mode=parser.add_mutually_exclusive_group()
mode.add_argument('--train',action='store_true')
mode.add_argument('--no-train',action='store_true')
args=parser.parse_args()
if args.train or args.no_train:
    nb.cells[1].source=nb.cells[1].source.replace('TRAIN = False','TRAIN = True') if args.train else nb.cells[1].source.replace('TRAIN = True','TRAIN = False')
try:
    NotebookClient(nb,timeout=1800,kernel_name='day4-project',resources={'metadata':{'path':str(path.parent)}}).execute()
finally:
    nbformat.write(nb,path)
print('Notebook completed:',path)

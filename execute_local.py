"""Execute every notebook code cell sequentially in a fresh process.

Captures actual stdout, DataFrame HTML and matplotlib PNG outputs without
requiring Jupyter's kernel transport on this Windows host.
"""
import os, sys, io, json, base64, contextlib, traceback, time
from pathlib import Path
import numpy, pandas, sklearn, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import torch
base=Path(__file__).resolve().parent
path=base/'notebook/neural_network.ipynb'
nb=json.loads(path.read_text(encoding='utf-8'))
if '--train' in sys.argv:
    nb['cells'][1]['source']=''.join(nb['cells'][1]['source']) if isinstance(nb['cells'][1]['source'],list) else nb['cells'][1]['source']
    nb['cells'][1]['source']=nb['cells'][1]['source'].replace('TRAIN = False','TRAIN = True')
if '--no-train' in sys.argv:
    nb['cells'][1]['source']=''.join(nb['cells'][1]['source']) if isinstance(nb['cells'][1]['source'],list) else nb['cells'][1]['source']
    nb['cells'][1]['source']=nb['cells'][1]['source'].replace('TRAIN = True','TRAIN = False')
os.chdir(path.parent)
namespace={'__name__':'__main__'}
outputs=[]
def display(obj):
    data={'text/plain':str(obj)}
    if hasattr(obj,'_repr_html_'):
        rendered=obj._repr_html_()
        if rendered:data['text/html']=rendered
    outputs.append({'output_type':'display_data','data':data,'metadata':{}})
def show(*args,**kwargs):
    for number in plt.get_fignums():
        fig=plt.figure(number);stream=io.BytesIO()
        fig.savefig(stream,format='png',dpi=110,bbox_inches='tight')
        outputs.append({'output_type':'display_data','data':{'image/png':base64.b64encode(stream.getvalue()).decode()},'metadata':{}})
    plt.close('all')
namespace['display']=display
plt.show=show
count=0
for index,cell in enumerate(nb['cells']):
    if cell['cell_type']!='code':continue
    count+=1; outputs=[];stream=io.StringIO();cell['execution_count']=count
    print(f'Executing code cell {count} (notebook cell {index})',flush=True)
    started=time.perf_counter()
    try:
        source=cell['source'] if isinstance(cell['source'],str) else ''.join(cell['source'])
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            exec(compile(source,f'notebook-cell-{index}','exec'),namespace)
    except Exception as exc:
        outputs.append({'output_type':'error','ename':type(exc).__name__,'evalue':str(exc),'traceback':traceback.format_exc().splitlines()})
        cell['outputs']=[{'output_type':'stream','name':'stdout','text':stream.getvalue()}]+outputs
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
        raise
    cell['outputs']=([{'output_type':'stream','name':'stdout','text':stream.getvalue()}] if stream.getvalue() else [])+outputs
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    print(stream.getvalue(),flush=True)
    print(f'Completed in {time.perf_counter()-started:.1f} s',flush=True)
print('ALL CELLS EXECUTED SUCCESSFULLY',flush=True)

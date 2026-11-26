"""Crop chart images and execute Python with state scoped to each question."""
import base64
from io import BytesIO
import json
from pathlib import Path
import selectors
import subprocess

from PIL import Image

TOOLS=[
 {'type':'function','function':{'name':'image_zoom_in_tool','description':'Crop and zoom an observed chart image. bbox_2d is [left, top, right, bottom] in coordinates normalized to 0..1000 on the selected image. img_idx is zero-based: original image 0, then images returned by tools. Choose a region containing the evidence needed for the question. This executes a real crop of the original pixels.',
  'parameters':{'type':'object','properties':{'bbox_2d':{'type':'array','items':{'type':'number'},'minItems':4,'maxItems':4},'label':{'type':'string'},'img_idx':{'type':'integer'}},'required':['bbox_2d','label','img_idx']}}},
 {'type':'function','function':{'name':'image_code_tool','description':'Execute real Python for chart calculations or image processing. code must be plain executable Python, without Markdown fences. With img_idx>=0, img is that observed image as a PIL.Image; use numpy.asarray(img) if an array is needed. img_idx=-1 supplies no image. NumPy, SciPy, pandas and Matplotlib can be imported. Variables persist within this question. Save generated PNGs under temp_output_dir; stdout and actual images are returned. No chart data table, plotting code or reference answer is supplied.',
  'parameters':{'type':'object','properties':{'code':{'type':'string'},'img_idx':{'type':'integer','default':-1}},'required':['code']}}}
]


def image_content(path):
    return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()}}


def compatible_observation(path):
    """Preserve raw pixels; pad only aspect ratios rejected by Qwen's processor."""
    path=Path(path)
    with Image.open(path) as raw:
        width,height=raw.size
        if max(width,height)<=200*min(width,height):return path,None
        size=(max(width,28,(height+199)//200),max(height,28,(width+199)//200))
        left,top=(size[0]-width)//2,(size[1]-height)//2
        canvas=Image.new('RGB',size,'white');canvas.paste(raw.convert('RGB'),(left,top))
        observed=path.with_name(path.stem+'-observation.png');canvas.save(observed)
    return observed,{'method':'white_padding_without_resampling','raw_image':str(path),
        'raw_size':[width,height],'returned_image':str(observed),'returned_size':list(size),
        'padding_ltrb':[left,top,size[0]-width-left,size[1]-height-top],
        'coordinate_space':'Later img_idx and bbox_2d refer to the returned padded canvas.'}


WORKER='''import contextlib,io,json,os,sys,traceback
from pathlib import Path
from PIL import Image
namespace={'__name__':'__main__','temp_output_dir':str(Path.cwd()/'outputs')}
Path(namespace['temp_output_dir']).mkdir(exist_ok=True)
for line in sys.stdin:
    request=json.loads(line)
    if request.get('image'):namespace['img']=Image.open(request['image'])
    else:namespace.pop('img',None)
    out,err=io.StringIO(),io.StringIO()
    status='success'
    try:
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):exec(compile(request['code'],'model_tool.py','exec'),namespace)
    except Exception:
        status='execution_error';traceback.print_exc(file=err)
    print(json.dumps({'status':status,'stdout':out.getvalue(),'stderr':err.getvalue()}),flush=True)
'''


class Environment:
    def __init__(self,images,root,python):
        self.images=[Path(p) for p in images]
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.python=python;self.process=None;self.index=0

    def execute(self,name,arguments):
        self.index+=1
        if name=='image_zoom_in_tool':
            bbox=arguments['bbox_2d'];idx=arguments['img_idx']
            if not (isinstance(idx,int) and 0<=idx<len(self.images)):raise ValueError('img_idx must reference an observed image')
            if len(bbox)!=4 or not (0<=bbox[0]<bbox[2]<=1000 and 0<=bbox[1]<bbox[3]<=1000):raise ValueError('bbox_2d must be valid coordinates normalized to 0..1000')
            image=Image.open(self.images[idx]);width,height=image.size
            rectangle=(int(bbox[0]*width/1000),int(bbox[1]*height/1000),int(bbox[2]*width/1000),int(bbox[3]*height/1000))
            if rectangle[2]<=rectangle[0] or rectangle[3]<=rectangle[1]:raise ValueError('Crop contains no original pixels')
            output=self.root/f'crop-{self.index}.png';image.crop(rectangle).save(output)
            observed,transform=compatible_observation(output);self.images.append(observed)
            with Image.open(observed) as crop:
                output_size=list(crop.size)
            result={'status':'success','name':name,'arguments':arguments,'source_image_index':idx,
                'source_size':[width,height],'source_bbox_pixels':list(rectangle),
                'output_image_index':len(self.images)-1,'images':[str(observed)],'output_size':output_size}
            if transform:result['observation_transforms']=[transform]
            return result
        if name!='image_code_tool':raise ValueError('Unknown tool')
        code=arguments['code']
        if code.lstrip().startswith('```'):raise ValueError('code must be Python without Markdown fences')
        idx=arguments.get('img_idx',-1)
        if not (isinstance(idx,int) and -1<=idx<len(self.images)):raise ValueError('img_idx must reference an observed image, or -1')
        output_dir=self.root/'outputs';output_dir.mkdir(exist_ok=True)
        before={p.name:p.stat().st_mtime_ns for p in output_dir.glob('*.png')}
        if self.process is None:
            import os
            environment={k:v for k,v in os.environ.items() if not any(token in k.upper() for token in ['API_KEY','TOKEN','SECRET'])}
            environment.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
            self.process=subprocess.Popen([self.python,'-u','-c',WORKER],cwd=self.root,env=environment,
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
        self.process.stdin.write(json.dumps({'code':code,'image':str(self.images[idx]) if idx>=0 else None})+'\n');self.process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout,selectors.EVENT_READ)
            ready=selector.select(30)
        if not ready:self.process.kill();self.process.wait();self.process=None;return {'status':'execution_error','name':name,'arguments':arguments,'stdout':'','stderr':'Python execution exceeded 30 seconds','images':[]}
        response=json.loads(self.process.stdout.readline())
        images=[];transforms=[]
        for path in output_dir.glob('*.png'):
            if before.get(path.name)!=path.stat().st_mtime_ns:
                target=self.root/f'python-{self.index}-{len(images)}.png';Image.open(path).save(target)
                observed,transform=compatible_observation(target)
                self.images.append(observed);images.append(str(observed))
                if transform:transforms.append(transform)
        result={**response,'name':name,'arguments':arguments,'images':images,'output_image_indices':list(range(len(self.images)-len(images),len(self.images)))}
        if transforms:result['observation_transforms']=transforms
        return result

    def close(self):
        if self.process:
            self.process.stdin.close()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            self.process.stdout.close()
            self.process.stderr.close()
            self.process=None


def observation_messages(call,result):
    text={k:v for k,v in result.items() if k!='images'}
    messages=[{'role':'tool','tool_call_id':call['id'],'content':json.dumps(text,ensure_ascii=False)}]
    if result.get('images'):
        messages.append({'role':'user','content':[{'type':'text','text':'The environment returned these actual tool images. Continue answering the original chart question using this observation and the original image.'},*[image_content(p) for p in result['images']]]})
    return messages

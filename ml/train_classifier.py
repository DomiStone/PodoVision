import os,json,glob,random
import pandas as pd, numpy as np, torch, torch.nn as nn
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score,classification_report,confusion_matrix,roc_auc_score
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
torch.manual_seed(42);np.random.seed(42);random.seed(42)
ROOT=os.environ.get("PAD_ROOT","data/pad"); meta=glob.glob(ROOT+"/**/metadata.csv",recursive=True)
if not meta: raise SystemExit("metadata.csv nicht gefunden")
d=pd.read_csv(meta[0]); image_col="img_id" if "img_id" in d.columns else "image_id"; label_col="diagnostic"; patient_col="patient_id"
paths={os.path.basename(p):p for p in glob.glob(ROOT+"/**/*.png",recursive=True)+glob.glob(ROOT+"/**/*.jpg",recursive=True)}
d["image_path"]=d[image_col].map(lambda x:paths.get(str(x),paths.get(str(x)+".png")))
d=d.dropna(subset=["image_path",label_col,patient_col]).copy()
classes=sorted(d[label_col].unique());ci={c:i for i,c in enumerate(classes)}
g=GroupShuffleSplit(1,test_size=.2,random_state=42);a,b=next(g.split(d,groups=d[patient_col]));tr=d.iloc[a];tmp=d.iloc[b]
g=GroupShuffleSplit(1,test_size=.5,random_state=43);a,b=next(g.split(tmp,groups=tmp[patient_col]));va=tmp.iloc[a];te=tmp.iloc[b]
aug=transforms.Compose([transforms.Resize((256,256)),transforms.RandomResizedCrop(224,scale=(.72,1)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(15),transforms.ColorJitter(.15,.15,.12,.04),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
plain=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
class DS(Dataset):
 def __init__(self,x,t):self.x=x.reset_index(drop=True);self.t=t
 def __len__(self):return len(self.x)
 def __getitem__(self,i):
  r=self.x.iloc[i];return self.t(Image.open(r.image_path).convert("RGB")),ci[r[label_col]]
def dl(x,t,s=False):return DataLoader(DS(x,t),32,shuffle=s,num_workers=2)
dev="cuda" if torch.cuda.is_available() else "cpu";m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT);m.classifier[1]=nn.Linear(m.classifier[1].in_features,len(classes));m.to(dev)
cnt=tr[label_col].value_counts();weights=torch.tensor([len(tr)/(len(classes)*cnt[c]) for c in classes],device=dev,dtype=torch.float)
loss=nn.CrossEntropyLoss(weight=weights,label_smoothing=.03);opt=torch.optim.AdamW(m.parameters(),2e-4,weight_decay=1e-4);best=-1
os.makedirs("web/model",exist_ok=True)
for ep in range(18):
 m.train()
 for x,y in dl(tr,aug,True):x,y=x.to(dev),y.to(dev);opt.zero_grad();l=loss(m(x),y);l.backward();opt.step()
 m.eval();yt=[];yp=[]
 with torch.no_grad():
  for x,y in dl(va,plain):yt+=y.tolist();yp+=m(x.to(dev)).argmax(1).cpu().tolist()
 score=balanced_accuracy_score(yt,yp);print("epoch",ep+1,"val_bal_acc",score)
 if score>best:best=score;torch.save(m.state_dict(),"web/model/classifier.pth")
m.load_state_dict(torch.load("web/model/classifier.pth",map_location=dev));m.eval();yt=[];prob=[]
with torch.no_grad():
 for x,y in dl(te,plain):yt+=y.tolist();prob+=torch.softmax(m(x.to(dev)),1).cpu().tolist()
pred=np.argmax(prob,1);report={"classes":classes,"balanced_accuracy":balanced_accuracy_score(yt,pred),"classification_report":classification_report(yt,pred,target_names=classes,output_dict=True,zero_division=0),"confusion_matrix":confusion_matrix(yt,pred).tolist()}
json.dump(report,open("web/model/metrics.json","w"),indent=2);json.dump(classes,open("web/model/classes.json","w"))
dummy=torch.randn(1,3,224,224,device=dev);torch.onnx.export(m,dummy,"web/model/classifier.onnx",input_names=["image"],output_names=["logits"],opset_version=17,dynamic_axes={"image":{0:"batch"}})
print(json.dumps(report,indent=2))

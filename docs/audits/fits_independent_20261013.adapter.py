import copy,json,sys
from types import SimpleNamespace
import numpy as np
import torch
import nullius_io as nio
sys.path.insert(0,'/opt/reference')
from models.FITS import Model

def train_forecaster(data,seed,smoke):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    length,horizon,batch=360,96,128
    args=SimpleNamespace(seq_len=length,pred_len=horizon,individual=False,enc_in=7,cut_freq=106)
    model=Model(args)
    windows=nio.forecast_data(length)
    xtr,ytr=windows['train_inputs'],windows['train_targets']
    # Dense checkpoint windows, all targets inside the visible validation prefix.
    series=np.concatenate([data['train'][-length:],data['val']])
    starts=np.arange(length,len(series)-horizon+1)
    xva=np.stack([series[s-length:s] for s in starts])
    yva=np.stack([series[s:s+horizon] for s in starts])
    if smoke:
        xtr,ytr,xva,yva=xtr[:8],ytr[:8],xva[:8],yva[:8]
        batch=8
    optimizer=torch.optim.Adam(model.parameters(),lr=.0005)
    best=float('inf')
    best_state=None
    bad=0
    records=[]
    metadata={'model':'FITS','train_windows':len(xtr),'checkpoint_windows':len(xva),'epochs':records,
              'fixed_parameters':{'length':360,'horizon':96,'cut_freq':106,'H_order':6,'base_T':24,
              'batch_cli':64,'effective_batch':128,'learning_rate':.0005,'epochs_max':50,
              'patience':20,'lradj':'type3','train_mode':1,'individual':False}}
    for epoch in range(1 if smoke else 50):
        model.train()
        order=torch.randperm(len(xtr)).numpy()
        losses=[]
        for start in range(0,len(order)-batch+1,batch):
            indices=order[start:start+batch]
            x,y=torch.tensor(xtr[indices]),torch.tensor(ytr[indices])
            optimizer.zero_grad()
            output,_=model(x)
            loss=((output[:,-horizon:]-y)**2).mean()
            if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        val_losses=[]
        with torch.no_grad():
            order=torch.randperm(len(xva)).numpy()
            for start in range(0,len(order)-batch+1,batch):
                indices=order[start:start+batch]
                output,_=model(torch.tensor(xva[indices]))
                val_losses.append(float(((output[:,-horizon:]-torch.tensor(yva[indices]))**2).mean()))
        value=float(np.mean(val_losses))
        if not np.isfinite(value): raise ValueError('Non-finite checkpoint validation')
        if value<=best:
            best,best_state,bad=value,copy.deepcopy(model.state_dict()),0
            metadata['best_epoch']=epoch+1
        else: bad+=1
        records.append({'epoch':epoch+1,'training_loss':float(np.mean(losses)),
                        'visible_validation_loss':value,'lr':optimizer.param_groups[0]['lr']})
        with open('/tmp/forecast-agent/training-summary.json','w') as stream: json.dump(metadata,stream)
        if bad>=20: break
        for group in optimizer.param_groups: group['lr']=.0005*(.95**epoch)
    model.load_state_dict(best_state)
    model.eval()
    def predict(history):
        with torch.no_grad():
            output,_=model(torch.tensor(history[None,-length:],dtype=torch.float32))
            return output[0,-horizon:].numpy()
    return predict

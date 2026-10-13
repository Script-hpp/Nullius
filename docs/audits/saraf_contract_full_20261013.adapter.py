import copy,json
import numpy as np
def take_batch(values, indices):
    """One-element/scalar indices must retain the batch axis, like DataLoader."""
    return values[np.asarray(indices,dtype=np.int64).reshape(-1)]

import sys, types, hashlib
import numpy as np
import torch
from types import SimpleNamespace
import nullius_io as nio
# Neither optional package is used by prepare_dataset/retrieve/encoder.
sys.modules['pandas'] = types.ModuleType('pandas')
progress = types.ModuleType('tqdm')
progress.tqdm = lambda items, **kw: items
sys.modules['tqdm'] = progress
sys.path.insert(0, '/opt/reference')
from models.SARAF import Model
sys.modules.pop('pandas',None)
sys.modules.pop('tqdm',None)

def train_forecaster(data, seed, smoke):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    windows = nio.forecast_data(720)
    class TrainDataset:
        pred_len, label_len = 96, 48
        def __len__(self): return len(windows['train_inputs'])
        def __getitem__(self, index):
            return (index, windows['train_inputs'][index], windows['train_targets'][index],
                    windows['train_calendar'][index], np.zeros((96,5),dtype=np.float32))
    args = SimpleNamespace(use_gpu=False,gpu=0,task_name='long_term_forecast',seq_len=720,
                           pred_len=96,features='M',enc_in=7,n_period=1,topm=20,freq='h',
                           time_aware_weight=.5,fusion_method='weighted')
    model = Model(args)
    model.rt.prepare_dataset(TrainDataset())
    if smoke:
        index = torch.arange(8)
        x = torch.tensor(windows['train_inputs'][:8])
        marks = torch.tensor(windows['train_calendar'][:8],dtype=torch.float32)
        y = torch.tensor(windows['train_targets'][:8])
        with torch.no_grad():
            retrieved = model.rt.retrieve(x,index,train=True,x_mark=marks)
        model.retrieval_temporal_dict = {'train':retrieved}
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        model.train()
        optimizer.zero_grad()
        loss = ((model(x,marks,index,mode='train')-y)**2).mean()
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        model.eval()
    else:
        def precompute(split):
            x,marks=windows[split+'_inputs'],windows[split+'_calendar']
            chunks=[]
            with torch.no_grad():
                for start in range(0,len(x),1024):
                    index=torch.arange(start,min(start+1024,len(x)))
                    chunks.append(model.rt.retrieve(torch.tensor(x[start:start+1024]),index,
                        train=split=='train',x_mark=torch.tensor(marks[start:start+1024],dtype=torch.float32)))
            return torch.cat(chunks,dim=1)
        model.retrieval_temporal_dict={'train':precompute('train'),'valid':precompute('val')}
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        best_loss=float('inf')
        best_state=None
        records=[]
        metadata={'train_windows':len(windows['train_inputs']),'checkpoint_windows':len(windows['val_inputs']),
                  'epochs':records,'fixed_parameters':{'batch_size':8,'max_epochs':10,'lr':.001,'lradj':'type1',
                  'seq_len':720,'pred_len':96,'n_period':1,'topm':20,'time_aware_weight':.5}}
        for epoch in range(10):
            model.train()
            order=torch.randperm(len(windows['train_inputs']))
            losses=[]
            for start in range(0,len(order),8):
                index=order[start:start+8]
                x=torch.tensor(take_batch(windows['train_inputs'],index))
                marks=torch.tensor(take_batch(windows['train_calendar'],index),dtype=torch.float32)
                y=torch.tensor(take_batch(windows['train_targets'],index))
                optimizer.zero_grad()
                loss=((model(x,marks,index,mode='train')-y)**2).mean()
                if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            val_losses=[]
            with torch.no_grad():
                # The original validation averages batch losses. No test data here.
                order=torch.randperm(len(windows['val_inputs']))
                for start in range(0,len(order),8):
                    index=order[start:start+8]
                    x=torch.tensor(take_batch(windows['val_inputs'],index))
                    marks=torch.tensor(take_batch(windows['val_calendar'],index),dtype=torch.float32)
                    y=torch.tensor(take_batch(windows['val_targets'],index))
                    loss=((model(x,marks,index,mode='valid')-y)**2).mean()
                    val_losses.append(float(loss))
            value=float(np.mean(val_losses))
            if not np.isfinite(value): raise ValueError('Non-finite checkpoint validation')
            if value<=best_loss:
                best_loss=value
                best_state=copy.deepcopy(model.state_dict())
                metadata['best_epoch']=epoch+1
            records.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),
                            'visible_validation_loss':value,'lr':optimizer.param_groups[0]['lr']})
            with open('/tmp/forecast-agent/training-summary.json','w') as stream:
                json.dump(metadata,stream)
            # Author type1 schedule is updated at each epoch end.
            for group in optimizer.param_groups: group['lr']=.001*(.5**epoch)
        model.load_state_dict(best_state)
        model.eval()
    retrieval_cache = {}
    def predict(history, *, calendar):
        assert calendar.shape==(720,5)
        x = torch.tensor(history[None],dtype=torch.float32)
        marks = torch.tensor(calendar[None],dtype=torch.float32)
        index = torch.tensor([0])
        with torch.no_grad():
            # Original SARAF samples retrieval once per window and stores it.
            # Lazy memoization keeps that property without exposing future windows.
            key = hashlib.sha256(history.tobytes()+calendar.tobytes()).digest()
            if key not in retrieval_cache:
                retrieval_cache[key] = model.rt.retrieve(x,index,train=False,x_mark=marks)
            model.retrieval_temporal_dict['test'] = retrieval_cache[key]
            return model(x,marks,index,mode='test').detach().numpy()[0]
    return predict

import csv
import datetime as dt

import numpy as np
import pytest

from repro_agent.harness import ett_data
from repro_agent.harness.selection_contract import forecast_public
from repro_agent.tasks import etth1_task, ett_task
from repro_agent.harness.forecast_worker import _accepts_calendar
from test_io_helper import _load_helper


@pytest.mark.parametrize('dataset,minutes,features', [('etth1',60,5),('ettm1',15,6)])
def test_preparation_preserves_real_calendar_without_changing_numeric_inputs(tmp_path, monkeypatch, dataset, minutes, features):
    monkeypatch.setattr(ett_data, 'borders', lambda name: (800,1200,1500))
    path = tmp_path / 'series.csv'
    dates = [dt.datetime(2016,7,1)+dt.timedelta(minutes=minutes*i) for i in range(1500)]
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['date',*[str(i) for i in range(7)]])
        writer.writerows([date.isoformat(sep=' '),*[i+j for j in range(7)]] for i,date in enumerate(dates))
    old,new = tmp_path/'old.npz',tmp_path/'new.npz'
    ett_data.prepare_public(dataset,path,old)
    monkeypatch.setenv('NULLIUS_FORECAST_CALENDAR','1')
    ett_data.prepare_public(dataset,path,new)
    with np.load(old) as legacy,np.load(new) as current:
        assert set(legacy.files)=={'train','val','test_inputs'}
        for key in legacy.files:
            assert np.array_equal(legacy[key],current[key])
        assert current['train_calendar'].shape==(800,features)
        expected=[dates[799].year,dates[799].month,dates[799].day,dates[799].weekday(),dates[799].hour]
        if features==6: expected.append(dates[799].minute)
        assert current['train_calendar'][799].tolist()==expected
        assert np.array_equal(current['test_calendar'][0,-1],current['val_calendar'][-1])
        assert len(current['test_calendar'])==len(current['test_inputs'])


def test_calendar_projection_and_training_windows_share_the_values_indices(tmp_path,monkeypatch):
    series=np.arange(1400*7,dtype=np.float32).reshape(1400,7)
    calendar=np.repeat(np.arange(1400)[:,None],5,axis=1)
    arrays={'train':series[:800],'val':series[800:], 'test_inputs':np.ones((3,720,7)),
            'train_calendar':calendar[:800],'val_calendar':calendar[800:], 'test_calendar':np.ones((3,720,5),dtype=np.int64)}
    public=forecast_public(arrays)
    assert public['val_calendar'].shape==(480,5) and public['test_calendar'].shape==(0,720,5)
    assert arrays['val_calendar'].shape==(600,5)
    helper=_load_helper(etth1_task().with_independent_selection(),tmp_path,monkeypatch,public,False)
    windows=helper.forecast_data(336,stride=4)
    assert np.array_equal(windows['train_calendar'][0],calendar[:336])
    assert np.array_equal(windows['val_calendar'][0],calendar[464:800])
    assert np.array_equal(windows['val_calendar'][1],calendar[488:824])


def test_new_product_task_requests_calendar_and_records_the_contract():
    legacy=etth1_task()
    current=legacy.with_independent_selection()
    assert not any('NULLIUS_FORECAST_CALENDAR' in cmd for cmd in legacy.setup_cmds)
    assert any('NULLIUS_FORECAST_CALENDAR=1 python /opt/prep/' in cmd for cmd in current.setup_cmds)
    assert 'calendar' in current.with_io_helper().data_description
    assert current.selection_info()['inference']=='causal_stream_v2'
    assert current.selection_info()['calendar_features']==['year','month','day','weekday','hour']
    assert ett_task('ettm1').with_independent_selection().selection_info()['calendar_features'][-1]=='minute'


def test_calendar_is_explicit_opt_in_and_legacy_wrappers_remain_compatible():
    def calendar_predictor(history, *, calendar): pass
    class LegacyWrapper:
        def __call__(self,*args,**kwargs): pass
    assert _accepts_calendar(calendar_predictor)
    assert not _accepts_calendar(lambda history: history)
    assert not _accepts_calendar(LegacyWrapper())
    assert not _accepts_calendar(np.negative)


def test_partial_calendar_metadata_is_rejected_before_worker_execution():
    arrays={'train':np.zeros((800,7)),'val':np.zeros((600,7)),
            'test_inputs':np.zeros((3,720,7)),'train_calendar':np.zeros((800,5))}
    with pytest.raises(ValueError,match='Incomplete'):
        forecast_public(arrays)

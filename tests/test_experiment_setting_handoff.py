"""Live FITS setup drift: source -> spec -> effective CONFIG, no provider calls."""
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from repro_agent import catalog, repo_source, spec_check
from repro_agent.setup_details import Doc
from repro_agent.spec import build_spec, verify_entries

ROOT = Path(__file__).resolve().parents[1]


def paper():
    return json.loads(gzip.decompress((ROOT / 'benchmark/source_reserves/2307.03756v3.json.gz').read_bytes()))['text']


def test_discovery_length_reaches_spec_even_when_extractor_omits_it():
    config, _ = catalog.build_paper(catalog.find_entry(catalog.load_catalog(), 'disc_2307_03756'))
    spec = build_spec(lambda *_: '[]', catalog.discovery_details(config, paper()),
                      [Doc('paper', paper())], 'FITS erreicht MSE 0.372', 'ETTh1', numeric_only=True)
    assert spec.get('seq_len').number == 360 and spec.get('seq_len').enforce
    code = 'CONFIG={"seq_len":90}\nlength=CONFIG["seq_len"]\n'
    fixed, report = spec_check.config_check(code, spec)
    namespace = {}; exec(fixed, namespace)
    assert namespace['length'] == 360 and report['overrides']


@pytest.mark.parametrize('value,quote,key', [
    ('90', r'& Look-back Window & \multicolumn{2}{c}{90} & \multicolumn{2}{c}{180} & \multicolumn{2}{c}{360}', 'seq_len'),
    ('32', r'% \textbf{Note that the following results are run under batch size 32 which may introduce sub-optimal results. }', 'batch_size'),
    ('180', 'We search input lengths [90, 180, 360, 720].', 'seq_len'),
])
def test_ambiguous_options_and_latex_comments_never_become_enforced_values(value, quote, key):
    source = r'\documentclass{article}' + '\nThe implementation follows the settings. ' + quote
    kept, dropped = verify_entries([{'key':key, 'value':value, 'quote':quote, 'source':'paper'}],
                                   [Doc('paper', source)], 'FITS', 'forecast', numeric_only=True)
    assert not kept and dropped


def test_unambiguous_setting_and_escaped_percent_stay_valid():
    text = r'\documentclass{article}' + '\n' + r'We train with batch size 128 and dropout 10\%. We use input length 360 and horizon 96.'
    raw = [{'key':'batch_size','value':'128','quote':'We train with batch size 128','source':'paper'},
           {'key':'seq_len','value':'360','quote':'input length 360 and horizon 96','source':'paper'}]
    kept, dropped = verify_entries(raw,[Doc('paper',text)],'FITS','forecast',numeric_only=True)
    assert len(kept)==2 and not dropped


def test_real_author_preset_overrides_wrong_live_extraction_and_derives_runtime_values():
    fixture=json.loads((ROOT/'tests/fixtures/fits_runtime_settings.json').read_text())
    docs=[Doc('paper',paper())]+[Doc('repo:'+p+'@d040bb0', t) for p,t in fixture['files'].items()]
    # Keep this about the parameter handoff, without importing unrelated trainers.
    docs=[Doc(d.name,d.text.replace('from exp.exp_main_F import Exp_Main','')) for d in docs]
    config,_=catalog.build_paper(catalog.find_entry(catalog.load_catalog(),'disc_2307_03756'))
    wrong=[{'key':'seq_len','value':'90','quote':r'& Look-back Window & \multicolumn{2}{c}{90} & \multicolumn{2}{c}{180} & \multicolumn{2}{c}{360} & \multicolumn{2}{c}{720} \\', 'source':'paper'}]
    spec=build_spec(lambda *_:json.dumps(wrong),catalog.discovery_details(config,paper()),docs,
                    'FITS erreicht 0.372 MSE auf ETTh1 (Horizont 96)','ETTh1',numeric_only=True,repo_listing=list(fixture['files']))
    values={i.key:i.number for i in spec.enforced()}
    assert values['seq_len']==360
    assert values['batch_size']==128  # command 64, effective runtime doubles without augmentation
    assert values['lr']==.0005 and values['epochs']==50 and values['patience']==20
    assert values['cut_freq']==106  # original CLI formula, not guessed from a score
    assert values['train_mode']==1
    assert 'dropout' not in values
    assert not {'hidden','layers','aug_rate','aug_data_size','label_len'} & values.keys()
    assert 'for seq_len in 360' in spec.excerpt and 'args.cut_freq = int(' in spec.excerpt
    sources={d.name:d.text for d in docs}
    from repro_agent.setup_details import quote_in_text
    from repro_agent.claims_source import _norm
    assert all(quote_in_text(i.quote,_norm(sources[i.source])) for i in spec.enforced())
    code='CONFIG={}\n'+ '\n'.join('x_'+k+'=CONFIG.get('+repr(k)+')' for k in values)
    fixed,report=spec_check.config_check(code,spec)
    namespace={};exec(fixed,namespace)
    assert namespace['x_seq_len']==360 and namespace['x_cut_freq']==106 and namespace['x_batch_size']==128
    assert not report['unused']


def test_forecast_reader_corrects_wrong_task_choice_and_loads_dataset_launch_script(monkeypatch,tmp_path):
    files={'README.md':'Run scripts with `_best` for the final reported results. Other scripts are ablations.',
           'AD/Toy.py':'class Model: pass\n','AD/main.py':'batch_size=16\n',
           'models/Toy.py':'class Model: pass\n',
           'scripts/Toy/etth1_best.sh':'python run_forecast.py --model Toy --data ETTh1 --seq_len 360 --pred_len 96\n',
           'run_forecast.py':'import argparse\nparser=argparse.ArgumentParser()\n',
           'scripts/Toy/etth1.sh':'python run_forecast.py --model Toy --data ETTh1 --seq_len 90 --pred_len 96\n'}
    class Sandbox:
        def base_snapshot(self,*a,**k):return 'snapshot'
        def branch(self,*a):return 'branch'
        def run(self,*a):
            data=('\0'.join(files)+'\0').encode()
            return SimpleNamespace(stdout=f'{len(data)}\n{data.count(0)}\n{hashlib.sha256(data).hexdigest()}\n',exit_code=0)
        def get_file(self,b,path):
            if path == '/tmp/.nullius_repo_listing': return ('\0'.join(files)+'\0').encode()
            return b'abc' if path.endswith('.head_commit') else files[path.removeprefix('/repo/')].encode()
    monkeypatch.setattr(repo_source,'select_files',lambda *_:['AD/Toy.py','AD/main.py','scripts/Toy/etth1.sh'])
    dump=repo_source.read_repo(Sandbox(),None,'https://github.com/example/toy','Toy erreicht MSE 0.4','ETTh1',
                              fixture_dir=tmp_path,unit_key='unit',model='Toy',mode='live')
    assert {'models/Toy.py','scripts/Toy/etth1_best.sh','run_forecast.py'} <= set(dump.files)
    assert not any(p.startswith('AD/') for p in dump.files)
    assert 'scripts/Toy/etth1.sh' not in dump.files


@pytest.mark.parametrize('mutation', [
    'def main():\n    args.batch_size *= 2\nmain()',
    'setattr(args, "batch_size", 128)',
    'for key, val in config.items():\n    setattr(args, key, val)',
    'args.__dict__.update(config)',
    'vars(args).update(config)',
    'args = load_config(args)',
])
def test_uninterpreted_namespace_mutations_never_preserve_a_false_default(mutation):
    from repro_agent.experiment_settings import python_settings
    source='import argparse\np=argparse.ArgumentParser()\np.add_argument("--batch_size",default=64)\nargs=p.parse_args()\n'+mutation
    values,_=python_settings(source,'python run.py')
    assert 'batch_size' not in values


def test_parse_known_args_and_config_handoff_are_not_treated_as_final_literals():
    from repro_agent.experiment_settings import python_settings
    text='p.add_argument("--batch_size",default=64)\nargs, rest = p.parse_known_args()\nargs = load_config(args)\n'
    assert 'batch_size' not in python_settings(text,'python run.py')[0]


def test_effective_namespace_name_and_known_branches_are_preserved():
    from repro_agent.experiment_settings import python_settings
    source='p.add_argument("--batch_size",default=32)\np.add_argument("--augment", action="store_true")\nopt=p.parse_args()\nif not opt.augment:\n    opt.batch_size *= 2\n'
    assert python_settings(source,'python run.py --batch_size 64')[0]['batch_size']==128
    assert python_settings(source,'python run.py --batch_size 64 --augment')[0]['batch_size']==64


def test_unknown_branch_and_unexecuted_authors_code_are_fail_closed(tmp_path):
    from repro_agent.experiment_settings import python_settings
    marker=tmp_path/'must_not_exist'
    source='p.add_argument("--batch_size",default=64)\nargs=p.parse_args()\n' + f'open({str(marker)!r},"w").write("bad")\n' + 'if external_condition():\n    args.batch_size=128\n'
    values,_=python_settings(source,'python run.py')
    assert 'batch_size' not in values and not marker.exists()


def test_shell_binding_is_sequential_and_does_not_guess_multivalue_or_conditional_choices():
    from repro_agent.experiment_settings import shell_calls
    calls=shell_calls('python run.py --seq_len $L\nL=360\npython run.py --seq_len $L\nfor L in 90 360\ndo\npython run.py --seq_len $L\ndone\nif external; then\nL=720\nfi\npython run.py --seq_len $L\n')
    assert '$L' in calls[0][1] and '--seq_len 360' in calls[1][1]
    assert '$L' in calls[2][1] and '$L' in calls[3][1]


def test_unresolved_active_model_parameter_prevents_experiment_configuration():
    from repro_agent.source_context import SourceContextBlocked
    docs=[Doc('paper','Input length 360.'),Doc('repo:models/Toy.py@a','class Model:\n    def __init__(self,configs):\n        self.cut_freq=configs.cut_freq\n'),
          Doc('repo:scripts/Toy/etth1.sh@a','python run.py --model Toy --data ETTh1 --pred_len 96\n'),
          Doc('repo:run.py@a','p.add_argument("--cut_freq",default=0)\nargs=p.parse_args()\nif external_condition():\n    args.cut_freq=106\n')]
    with pytest.raises(SourceContextBlocked,match='unresolved model'):
        build_spec(lambda *_:'[]',[],docs,'Toy erreicht MSE 0.4','ETTh1',numeric_only=True)


def test_unresolved_device_flag_does_not_block_a_cpu_launch():
    """SARAF run.py derives args.gpu from device_ids; the model reads configs.gpu. A device choice is not a model setting."""
    docs=[Doc('paper','Input length 720.'),
          Doc('repo:models/Toy.py@a','class Model:\n    def __init__(self,configs):\n        self.k=configs.topm\n        self.dev=configs.gpu\n'),
          Doc('repo:scripts/Toy/etth1.sh@a','python run.py --model Toy --data ETTh1 --pred_len 96 --seq_len 720 --topm 20\n'),
          Doc('repo:run.py@a','p.add_argument("--gpu",type=int,default=0)\np.add_argument("--topm",type=int,default=5)\n'
              'p.add_argument("--seq_len",type=int,default=96)\np.add_argument("--pred_len",type=int,default=96)\n'
              'p.add_argument("--model",default="x")\np.add_argument("--data",default="x")\n'
              'args=p.parse_args()\nif args.use_multi_gpu:\n    args.gpu=args.device_ids[0]\n')]
    spec=build_spec(lambda *_:'[]',[],docs,'Toy erreicht MSE 0.4','ETTh1',numeric_only=True)
    assert spec.get('seq_len').number==720 and spec.get('gpu') is None


def test_unknown_identity_and_changed_horizon_cannot_be_silently_used():
    from repro_agent.source_context import SourceContextBlocked
    model=Doc('repo:models/Toy.py@a','class Model:\n    pass\n')
    script=Doc('repo:run.py@a','p.add_argument("--model",default="Toy")\np.add_argument("--pred_len",default=96)\nargs=p.parse_args()\nargs.pred_len=192\n')
    sh=Doc('repo:scripts/Toy/etth1.sh@a','python run.py --model Toy --pred_len 96\n')
    with pytest.raises(SourceContextBlocked,match='horizon'):
        build_spec(lambda *_:'[]',[],[model,script,sh],'Toy erreicht MSE 0.4','ETTh1',numeric_only=True)


def test_unused_mandatory_setting_is_blocked_before_sandbox_execution():
    from repro_agent.loop import Orchestrator
    from repro_agent.spec import Spec, SpecItem
    o=Orchestrator.__new__(Orchestrator)
    o.sandbox=object()  # any sandbox call would fail this regression
    o.allow_test_feedback=False; o.independent_selection=True
    o._active_spec=lambda:Spec('Toy','forecast',[SpecItem('cut_freq','106','q','paper','paper',106,enforce=True)])
    result=o._run_and_eval(None,"CONFIG={'cut_freq':106}\n",0,None,'mse')
    assert result['status']=='ungueltig' and 'cut_freq' in result['detail']


def test_final_admission_never_turns_partial_settings_into_a_paper_verdict():
    from repro_agent.loop import Orchestrator
    from repro_agent.method_guard import MethodImplementationBlocked
    o=Orchestrator.__new__(Orchestrator)
    o.independent_selection=True; o._current_spec_score=lambda _: (1,2)
    with pytest.raises(MethodImplementationBlocked,match='mandatory experiment settings'):
        o._admit_final_code(None,None,None,None,'code',0)


@pytest.mark.parametrize('mutation',[
    'hp=args\nhp.batch_size=256\n',
    'class Exp:\n    def __init__(self,args):\n        self.args=args\n        self.args.batch_size=256\nExp(args)\n',
])
def test_namespace_alias_or_trainer_attribute_cannot_preserve_a_false_cli_value(mutation):
    from repro_agent.experiment_settings import python_settings
    source='p.add_argument("--batch_size",default=32)\nargs=p.parse_args()\n'+mutation
    values=python_settings(source,'python run.py --batch_size 64')[0]
    assert values.get('batch_size') in (None,256)  # exact module-level alias mutation may be derived


def test_imported_trainer_mutation_cannot_preserve_the_entrypoint_cli_value():
    from repro_agent.experiment_settings import python_settings
    source='p.add_argument("--batch_size",default=32)\nargs=p.parse_args()\n'
    trainer='class Exp:\n    def __init__(self,args):\n        self.args=args\n        self.args.batch_size*=4\n'
    assert 'batch_size' not in python_settings(source,'python run.py --batch_size 64',[trainer])[0]
    pointer_only='class Data:\n    def __init__(self,config):\n        self.args=config\n'
    assert python_settings(source,'python run.py --batch_size 64',[pointer_only])[0]['batch_size']==64


def test_identity_loops_select_the_requested_members_and_not_a_parameter_grid():
    from repro_agent.experiment_settings import shell_calls
    source='for model in Toy Other\ndo\nfor data in ETTh1 ETTh2\ndo\nfor pred_len in 96 192\ndo\nfor L in 90 360\ndo\npython run.py --model $model --data $data --pred_len $pred_len --seq_len $L\n'
    call=shell_calls(source,{'model':'Toy','data':'ETTh1','pred_len':96})[0][1]
    assert '--model Toy' in call and '--data ETTh1' in call and '--pred_len 96' in call
    assert '--seq_len $L' in call
    assert shell_calls(source,{'pred_len':720})==[]


@pytest.mark.parametrize('model,extra,expected',[('SparseTSF','period_len',24),('SegRNN','seg_len',24)])
def test_real_cross_paper_launch_scripts_select_horizon_96_and_length_720(model,extra,expected):
    from repro_agent.spec import runtime_items
    scripts=json.loads((ROOT/'tests/fixtures/forecast_identity_loops.json').read_text())
    src,text=next((key.split(' ',1)[1],text) for key,text in scripts.items() if key.startswith(model+' '))
    entry='import argparse\np=argparse.ArgumentParser()\n'+''.join(
        f'p.add_argument("--{key}",default={value!r})\n' for key,value in
        [('model',model),('data','ETTh1'),('seq_len',96),('pred_len',96),('learning_rate',.001),('batch_size',32),('train_epochs',10),('patience',3),(extra,48)])+'args=p.parse_args()\n'
    core=f'class Model:\n    def __init__(self,configs):\n        self.seq_len=configs.seq_len\n        self.pred_len=configs.pred_len\n        self.extra=configs.{extra}\n'
    docs=[Doc(src,text),Doc('repo:run_longExp.py@a',entry),Doc(f'repo:models/{model}.py@a',core)]
    # A separate ablation cannot displace the source-bound 720 setting.
    docs.append(Doc(f'repo:scripts/{model}/Lookback_96/etth1.sh@a',f'python run_longExp.py --model {model} --data ETTh1 --seq_len 96 --pred_len 96\n'))
    values={i.key:i.number for i in runtime_items(docs,[d.name for d in docs],model,'forecast','ETTh1',selected_length=720)}
    assert values['seq_len']==720 and values['pred_len']==96 and values[extra]==expected


def test_unresolved_effective_optimizer_parameter_cannot_fall_back_to_raw_cli():
    from repro_agent.source_context import SourceContextBlocked
    core=Doc('repo:models/Toy.py@a','class Model:\n    pass\n')
    entry=Doc('repo:run.py@a','p.add_argument("--learning_rate",default=.001)\nargs=p.parse_args()\ndef main():\n    args.learning_rate*=2\nmain()\n')
    script=Doc('repo:scripts/Toy/etth1.sh@a','python run.py --learning_rate .001\n')
    with pytest.raises(SourceContextBlocked,match='unresolved model parameters'):
        build_spec(lambda *_:'[]',[],[core,entry,script],'Toy erreicht MSE 0.4','ETTh1',numeric_only=True)


def test_multivariate_identity_excludes_univariate_argparse_abbreviation():
    from repro_agent.spec import runtime_items
    entry=Doc('repo:run.py@a','p.add_argument("--features",default="M")\np.add_argument("--enc_in",default=7)\np.add_argument("--seq_len",default=360)\nargs=p.parse_args()\n')
    docs=[entry,Doc('repo:models/Toy.py@a','class Model:\n    pass\n'),
          Doc('repo:scripts/Toy/etth1.sh@a','python run.py --model Toy --data ETTh1 --pred_len 96 --features M --seq_len 360\n'),
          Doc('repo:scripts/Toy/univariate/etth1.sh@a','python run.py --model Toy --data ETTh1 --pred_len 96 --feature S --enc_in 1 --seq_len 720\n')]
    items=runtime_items(docs,[d.name for d in docs],'Toy','forecast','ETTh1')
    assert {i.number for i in items if i.key=='seq_len'}=={360}
    from repro_agent.experiment_settings import python_settings
    assert python_settings(entry.text,'python run.py --feature S')[0]['features']=='S'


@pytest.mark.parametrize('field,value',[('features','S'),('enc_in',1)])
def test_unknown_feature_mode_is_not_silently_assumed_multivariate(field,value):
    from repro_agent.source_context import SourceContextBlocked
    entry=Doc('repo:run.py@a','p.add_argument("--features",default="M")\np.add_argument("--enc_in",default=7)\np.add_argument("--seq_len",default=360)\nargs=p.parse_args()\ndef main():\n'+f'    args.{field}={value!r}\n'+'main()\n')
    docs=[entry,Doc('repo:models/Toy.py@a','class Model:\n    pass\n'),
          Doc('repo:scripts/Toy/etth1.sh@a','python run.py --model Toy --data ETTh1 --pred_len 96 --features M\n')]
    with pytest.raises(SourceContextBlocked,match='feature'):
        build_spec(lambda *_:'[]',[],docs,'Toy erreicht MSE 0.4','ETTh1',numeric_only=True)


@pytest.mark.parametrize('field,value',[('features','S'),('enc_in',1)])
@pytest.mark.parametrize('post_parse',[True,False])
def test_effective_univariate_settings_never_fall_back_to_raw_cli(field,value,post_parse):
    from repro_agent.source_context import SourceContextBlocked
    declarations={'features':'M','enc_in':7}
    if not post_parse:declarations[field]=value
    text=''.join(f'p.add_argument("--{key}",default={default!r})\n' for key,default in declarations.items())+'args=p.parse_args()\n'
    if post_parse:text+=f'args.{field}={value!r}\n'
    docs=[Doc('repo:run.py@a',text),Doc('repo:models/Toy.py@a','class Model:\n    pass\n'),
          Doc('repo:scripts/Toy/etth1.sh@a','python run.py --model Toy --data ETTh1 --pred_len 96 --learning_rate .01 --batch_size 64 --seq_len 336\n')]
    with pytest.raises(SourceContextBlocked,match='feature/channel identity'):
        build_spec(lambda *_:'[]',[],docs,'Toy erreicht MSE 0.4','ETTh1',numeric_only=True)


def test_legacy_raw_rules_and_default_stripping_respect_resolved_model_and_features():
    from repro_agent.spec import raw_items,drop_run_script_defaults
    entry=Doc('repo:run.py@a','p.add_argument("--learning_rate",default=.001)\n')
    sh=Doc('repo:scripts/Linear/etth1.sh@a','model_name=DLinear\npython run.py --model $model_name --data ETTh1 --pred_len 96 --learning_rate .005\npython run.py --model NLinear --data ETTh1 --pred_len 96 --feature S --learning_rate .006\n')
    docs=[entry,sh]
    assert not [i for i in raw_items(docs,[d.name for d in docs],'NLinear','forecast','ETTh1',96) if i.key=='lr' and i.number in (.005,.006)]
    excerpt=''.join(f'=== {d.name} ===\n{d.text}' for d in docs)
    unchanged,dropped=drop_run_script_defaults(excerpt,'ETTh1',96,'NLinear')
    assert unchanged==excerpt and not dropped


def test_equal_rank_author_presets_stop_instead_of_silently_ignoring_settings():
    from repro_agent.source_context import SourceContextBlocked
    entry=Doc('repo:run.py@a','p.add_argument("--seq_len",default=360)\nargs=p.parse_args()\n')
    docs=[entry,Doc('repo:models/Toy.py@a','class Model:\n    pass\n')]+[
        Doc(f'repo:scripts/Toy/etth1_{length}.sh@a',f'python run.py --model Toy --data ETTh1 --pred_len 96 --seq_len {length}\n') for length in (360,720)]
    with pytest.raises(SourceContextBlocked,match='ambiguous'):
        build_spec(lambda *_:'[]',[],docs,'Toy erreicht MSE 0.4','ETTh1',numeric_only=True)


def test_flag_relabelled_by_extractor_is_not_enforced_twice():
    """SARAF: the extractor mapped --n_period to 'layers'; the launch already resolves n_period."""
    docs=[Doc('paper','Input length 720.'),
          Doc('repo:models/Toy.py@a','class Model:\n    def __init__(self,configs):\n        self.p=configs.n_period\n'),
          Doc('repo:scripts/Toy/etth1.sh@a','python run.py --model Toy --data ETTh1 --pred_len 96 --seq_len 720\n'),
          Doc('repo:run.py@a',"p.add_argument('--n_period', type=int, default=1, help='Number of Periods')\n"
              'p.add_argument("--seq_len",type=int,default=96)\np.add_argument("--pred_len",type=int,default=96)\n'
              'p.add_argument("--model",default="x")\np.add_argument("--data",default="x")\nargs=p.parse_args()\n')]
    extractor=json.dumps([{'key':'layers','value':'1','source':'repo:run.py@a',
                           'quote':"p.add_argument('--n_period', type=int, default=1, help='Number of Periods')"}])
    spec=build_spec(lambda *_:extractor,[],docs,'Toy erreicht MSE 0.4','ETTh1',numeric_only=True)
    assert spec.get('n_period').number==1 and spec.get('layers') is None

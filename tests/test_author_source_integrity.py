"""Reference completeness checks: synthetic repositories, no APIs or training."""
import json
import pytest
from repro_agent.source_context import author_excerpt, SourceContextBlocked
from repro_agent.spec import build_spec
from repro_agent.setup_details import Doc, SetupDetailV2, scope_exclusion


def test_author_context_keeps_forward_after_large_diagnostic_preamble_and_deep_dependency():
    # Previously the first 6000 chars of the model, and a prefix of the imported
    # layer, could contain declarations but none of the actual computation.
    docs = [Doc('repo:models/NewForecast.py@abc', '# diagnostic output\n' * 400 +
                'from layers.retriever import RetrievalTool\nclass Model:\n    def forward(self, x):\n        return x - x[-1]\n'),
            Doc('repo:layers/retriever.py@abc', '# stationarity helpers\n' * 1600 +
                'from .distance import distance\nclass RetrievalTool:\n    def retrieve(self, x):\n        return distance(x)\n'),
            Doc('repo:layers/distance.py@abc', 'def distance(x):\n    return x.sum()\n'),
            Doc('repo:run.py@abc', "parser.add_argument('--n_period', default=1)\n"),
            Doc('repo:exp/train.py@abc', 'def train():\n    loss.backward()\n    adjust_learning_rate(optimizer)\n')]
    ex, names = author_excerpt(docs, 'NewForecast', 'ETTh1')
    assert 'def forward' in ex and 'def retrieve' in ex and 'def distance' in ex
    assert "default=1" in ex and 'adjust_learning_rate(optimizer)' in ex
    assert len(names) == 5
    spec = build_spec(lambda *_: '[]', [], docs, 'NewForecast erreicht MSE 0.4', 'ETTh1', numeric_only=True)
    assert 'def forward' in spec.excerpt and 'def retrieve' in spec.excerpt


@pytest.mark.parametrize('failure', ['syntax', 'missing_dependency', 'budget'])
def test_incomplete_reference_stops_before_extraction_or_training(failure):
    text = 'from layers.distance import distance\nclass Model:\n    def forward(self, x):\n        return distance(x)\n'
    docs = [Doc('repo:models/NewForecast.py@c', text)]
    listing = ['models/NewForecast.py', 'layers/distance.py']
    if failure == 'syntax':
        docs[0] = Doc(docs[0].name, text + 'def broken(')
    if failure != 'missing_dependency':
        docs += [Doc('repo:layers/distance.py@c', 'def distance(x):\n    return x\n')]
    with pytest.raises(SourceContextBlocked):
        author_excerpt(docs, 'NewForecast', 'ETTh1', listing=listing, budget=20 if failure == 'budget' else 160000)


def test_budget_block_prevents_numerical_model_call():
    docs = [Doc('repo:models/NewForecast.py@c', '# huge\n' * 30000 + 'class Model:\n    pass\n')]
    with pytest.raises(SourceContextBlocked):
        build_spec(lambda *_: pytest.fail('must stop before model call'), [], docs,
                   'NewForecast erreicht MSE 0.4', 'ETTh1', numeric_only=True)


@pytest.mark.parametrize('filename', ['exp/exp_long_term_forecasting.py', 'layers/Retrieval.py', 'utils/early_stopping.py'])
def test_shared_training_and_algorithm_files_are_not_mistaken_for_foreign_models(filename):
    d = SetupDetailV2('schedule', 'type1', 'q', 'repo:'+filename+'@c',
                      applies_to={'model': 'all', 'dataset': 'all', 'setting': 'not stated'})
    paper = 'We use Long Term Forecasting. Long Term Forecasting uses Retrieval. Retrieval improves Forecasting.'
    assert scope_exclusion(d, 'NewForecast erreicht MSE 0.4', 'ETTh1', paper) == ''


def test_foreign_model_scope_still_rejected():
    d = SetupDetailV2('dropout', '0.2', 'q', 'repo:models/DLinear.py@c', applies_to={'model': 'all'})
    assert scope_exclusion(d, 'NLinear erreicht MSE 0.4', 'ETTh1', 'DLinear and NLinear are evaluated.').startswith('other model')


def test_source_dependencies_use_exact_paths_and_preserve_data_loader_code():
    docs=[Doc('repo:models/NewForecast.py@c','import layers.first, layers.second as second\nfrom data.loader import Loader\nclass Model:\n    pass\n'),
          Doc('repo:layers/first.py@c','VALUE=1\n'),Doc('repo:layers/second.py@c','VALUE=2\n'),
          Doc('repo:unrelated/second.py@c','WRONG_MODULE=True\n'),Doc('repo:data/loader.py@c','class Loader:\n    pass\n')]
    ex,names=author_excerpt(docs,'NewForecast','ETTh1')
    assert 'VALUE=2' in ex and 'class Loader' in ex and 'WRONG_MODULE' not in ex

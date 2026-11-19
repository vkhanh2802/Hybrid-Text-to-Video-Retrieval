"""Offline local text planner test using production prompt/schema/normalizer.

No model downloads, HTTP API calls, retrieval, DRES or active-setting changes.
"""
import json
import os
from pathlib import Path
from time import perf_counter

from tools.benchmark_user_five_ocr import ROOT, save, sha


def run(output, model_path, device='cpu'):
    wall_start = perf_counter()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    import numpy
    import torch
    import jsonschema
    import psutil
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kis.llm import HTTPQueryStructurer, QueryAPIConfig, parse_query_json
    from kis.structure_query import (
        SYSTEM_INSTRUCTION, QUERY_RESPONSE_SCHEMA, QUERY_RESPONSE_EXAMPLE,
        create_structured_query,
    )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    spec = ROOT / 'benchmarks/local_query_planner_20261118.json'
    data = json.loads(spec.read_text(encoding='utf-8'))
    base = ROOT / data['base_queries']
    queries = json.loads(base.read_text(encoding='utf-8'))['queries'] + data['probes']
    pins = {p.relative_to(ROOT).as_posix(): sha(p) for p in [
        spec, base, Path(__file__).resolve(), ROOT / 'kis/llm.py',
        ROOT / 'kis/structure_query.py', ROOT / 'config/pipeline.json',
        ROOT / 'config/school-runtime.local.json', ROOT / 'config/models.json']}
    path = Path(model_path).resolve()
    assert path.is_dir()
    model_type = json.loads((path / 'config.json').read_text(encoding='utf-8'))['model_type']
    start = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if model_type == 'qwen2_5_vl':
        assert device == 'cuda:0', 'Cached VL model is tested only as quantized GPU text inference'
        from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
        torch.set_num_threads(1)
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                path, local_files_only=True, low_cpu_mem_usage=True, device_map={'': 0},
                attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)).eval()
        finally:
            torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=torch.float32 if device == 'cpu' else torch.float16,
            attn_implementation='sdpa').to(device).eval()
    load_seconds = perf_counter() - start
    config = QueryAPIConfig(provider='ollama', protocol='openai', model=path.name,
                            base_url='http://127.0.0.1:0/v1', response_format='json_object',
                            temperature=0, max_tokens=1024)

    class LocalGenerator(HTTPQueryStructurer):
        """Only the transport is overridden: production constructs the prompt."""
        def _post(self, route, payload):
            assert route == '/chat/completions'
            self.raw = None
            text = tokenizer.apply_chat_template(payload['messages'], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors='pt').to(device)
            self.input_tokens = inputs['input_ids'].shape[1]
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            start = perf_counter()
            with torch.inference_mode():
                result = model.generate(**inputs, max_new_tokens=config.max_tokens,
                                        do_sample=False, use_cache=True,
                                        pad_token_id=tokenizer.eos_token_id)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            self.inference_seconds = perf_counter() - start
            tokens = result[0, self.input_tokens:]
            self.output_tokens = len(tokens)
            self.raw = tokenizer.decode(tokens, skip_special_tokens=True)
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            self.truncated = len(tokens) >= config.max_tokens and int(tokens[-1]) not in eos
            return {'choices': [{'message': {'content': self.raw},
                                 'finish_reason': 'length' if self.truncated else 'stop'}]}

    generator = LocalGenerator(config, SYSTEM_INSTRUCTION, QUERY_RESPONSE_SCHEMA, example=QUERY_RESPONSE_EXAMPLE)
    protocol = {'model_snapshot': str(path), 'model_name': path.parent.parent.name,
                'device': device, 'dtype': str(model.dtype), 'threads': 4,
                'quantization': 'NF4 double-quantized' if model_type == 'qwen2_5_vl' else None,
                'input_modality': 'text_only_no_images',
                'transport': 'in_process_transformers_no_HTTP', 'internet_calls': False,
                'local_files_only': True, 'production_prompt_and_normalizer': True,
                'json_grammar_enforced': False, 'manual_repair': False, 'repetitions': 1,
                'max_new_tokens': config.max_tokens, 'queries': queries, 'source_sha256': pins,
                'model_load_seconds': load_seconds, 'GPU_used_for_CPU_test': device != 'cpu',
                'retrieval_run': False, 'reranker_run': False, 'active_settings_changed': False}
    save(output / 'PROTOCOL.json', protocol)
    print(json.dumps({'phase': 'model_loaded', 'seconds': load_seconds, 'device': device}), flush=True)
    rows = []
    for q in queries:
        start = perf_counter()
        normalized, error = None, None
        try:
            normalized = create_structured_query(q['query'], query_id=q['id'], output_path=None, generator=generator)
        except (ValueError, RuntimeError) as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)[:1000]}
        planner_seconds = perf_counter() - start
        parsed, schema_error = None, None
        try:
            parsed = parse_query_json(generator.raw)
            jsonschema.validate(parsed, QUERY_RESPONSE_SCHEMA)
            schema_valid = True
        except (ValueError, RuntimeError, jsonschema.ValidationError) as exc:
            schema_valid = False
            schema_error = str(exc).splitlines()[0][:1000]
        row = {**q, 'planner_seconds': planner_seconds, 'inference_seconds': generator.inference_seconds,
               'input_tokens': generator.input_tokens, 'output_tokens': generator.output_tokens,
               'output_tokens_per_second': generator.output_tokens / generator.inference_seconds,
               'truncated': generator.truncated, 'raw_schema_valid': schema_valid,
               'normalized_valid': normalized is not None, 'error': error, 'schema_error': schema_error,
               'raw_output': generator.raw, 'parsed': parsed, 'structured': normalized,
               'process_RSS_bytes': psutil.Process().memory_info().rss,
               'GPU_peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0}
        save(output / (q['id'] + '.json'), row)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ['id', 'planner_seconds', 'output_tokens',
                                             'raw_schema_valid', 'normalized_valid', 'truncated', 'error']}), flush=True)
    import statistics
    assert all(sha(ROOT / p) == h for p, h in pins.items())
    report = {'status': 'completed', 'model_snapshot': str(path), 'device': device,
              'queries': len(rows), 'raw_schema_valid': sum(r['raw_schema_valid'] for r in rows),
              'normalized_valid': sum(r['normalized_valid'] for r in rows),
              'model_load_seconds': load_seconds, 'actual_wall_seconds': perf_counter() - wall_start,
              'timing': {k: fn(r['planner_seconds'] for r in rows)
                         for k, fn in [('median', statistics.median), ('mean', statistics.mean), ('min', min), ('max', max)]},
              'max_process_RSS_bytes': max(r['process_RSS_bytes'] for r in rows),
              'GPU_peak_allocated_bytes': max(r['GPU_peak_allocated_bytes'] for r in rows),
              'source_and_active_settings_unchanged': True,
              'semantic_faithfulness': 'requires manual review; schema success is not semantic correctness',
              'runs': rows}
    save(output / 'REPORT.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}), flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda:0'])
    run(**vars(p.parse_args()))

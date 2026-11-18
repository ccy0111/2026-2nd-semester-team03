"""Create a separate decoder export that computes only the final-position logits.

Supports the verified WebScore FP32 export contract. Input/weights are unchanged;
outputs include an optimized FP32 decoder and optional freshly quantized INT8.
Never modifies the source export. Requires onnx and (for INT8) onnxruntime.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import onnx
from onnx import helper, numpy_helper


def optimize(source, destination, *, quantize=True):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError('Use a new output directory; existing exports are never overwritten')
    model = onnx.load(source / 'decoder_kv.onnx')
    producers = [n for n in model.graph.node if 'logits' in n.output]
    if len(producers) != 1 or producers[0].op_type != 'MatMul' or len(producers[0].input) != 2:
        raise ValueError('Unsupported export: logits must be produced by the final MatMul')
    linear = producers[0]
    norm = [n for n in model.graph.node if linear.input[0] in n.output]
    if len(norm) != 1 or norm[0].op_type != 'LayerNormalization':
        raise ValueError('Unsupported export: expected final LayerNormalization before logits')
    existing = {x for n in model.graph.node for x in (*n.input, *n.output)}
    prefix = 'webscore_last_logits'
    if any(x.startswith(prefix) for x in existing):
        raise ValueError('Export is already optimized or names conflict')
    for name, values in [('starts', [-1]), ('ends', [np.iinfo(np.int64).max]), ('axes', [1])]:
        model.graph.initializer.append(numpy_helper.from_array(np.array(values, np.int64), f'{prefix}_{name}'))
    # Slice before final norm AND output projection. Attention/KV remain untouched.
    node = helper.make_node('Slice', [norm[0].input[0], f'{prefix}_starts', f'{prefix}_ends', f'{prefix}_axes'], [prefix], name=prefix)
    norm[0].input[0] = prefix
    idx = list(model.graph.node).index(norm[0])
    model.graph.node.insert(idx, node)
    output = next(o for o in model.graph.output if o.name == 'logits')
    output.type.tensor_type.shape.dim[1].ClearField('dim_param')
    output.type.tensor_type.shape.dim[1].dim_value = 1
    onnx.checker.check_model(model)
    destination.mkdir(parents=True)
    for name in ('config.json','tokenizer.json','mel_encoder.onnx','token_embed.onnx'):
        shutil.copyfile(source/name, destination/name)
    cfg = json.loads((destination/'config.json').read_text())
    cfg['decoder_logits'] = 'last'
    (destination/'config.json').write_text(json.dumps(cfg, indent=2))
    onnx.save_model(model, destination/'decoder_kv.onnx')
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(destination/'decoder_kv.onnx'), str(destination/'decoder_kv.int8.onnx'), weight_type=QuantType.QInt8)
    return destination

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source');p.add_argument('destination');p.add_argument('--no-int8',action='store_true')
    a=p.parse_args(); print(optimize(a.source,a.destination,quantize=not a.no_int8))

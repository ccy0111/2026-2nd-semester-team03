import argparse
from pathlib import Path
import onnx
from onnx import helper as h,TensorProto as T
p=argparse.ArgumentParser(description="Specialize shape-only operands to shared host inputs")
p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.input.resolve()==a.output.resolve(): raise ValueError('Output must not overwrite the source graph')
m=onnx.load(a.input)
replacements={'Reshape':'shape_q','Reshape_1':'shape_kv','Reshape_2':'shape_kv','Expand':'shape_expand','Expand_1':'shape_expand','Reshape_4':'shape_expanded_kv','Reshape_6':'shape_expanded_kv','Reshape_7':'shape_out'}
onnx.checker.check_model(m)
layer_count=next(v for v in m.graph.input if v.name=='past_k').type.tensor_type.shape.dim[0].dim_value
if layer_count <= 0: raise ValueError('Expected a fixed positive decoder layer count')
expected={f'dec//layers.{i}/self_attn/{op}' for i in range(layer_count) for op in replacements}
actual={n.name for n in m.graph.node if '/self_attn/' in n.name and n.name.rsplit('/',1)[-1] in replacements}
if actual != expected: raise ValueError('Unrecognized attention shape graph; refusing partial rewrite')
if sum(n.op_type=='Range' and n.name=='dec//Range' for n in m.graph.node)!=1:
 raise ValueError('Expected exactly one position Range')
count=0
for n in m.graph.node:
 if '/self_attn/' in n.name and n.name.rsplit('/',1)[-1] in replacements:
  if n.op_type not in ('Reshape','Expand') or len(n.input)!=2: raise ValueError('Unexpected shape consumer')
  n.input[1]=replacements[n.name.rsplit('/',1)[-1]];count+=1
for name,size in [('shape_q',4),('shape_kv',4),('shape_expand',5),('shape_expanded_kv',4),('shape_out',3)]:m.graph.input.append(h.make_tensor_value_info(name,T.INT64,[size]))
# Host arange(0, seq) replaces Range, retaining FP16 downstream cast if present.
for n in m.graph.node:
 if n.op_type=='Range' and n.name=='dec//Range':
  n.op_type='Identity';del n.input[:];n.input.append('position_range')
m.graph.input.append(h.make_tensor_value_info('position_range',T.FLOAT,['seq']))
# Reverse reachable prune, not graph optimization/weight arithmetic transformation.
needed={o.name for o in m.graph.output};keep=[]
for n in reversed(m.graph.node):
 if any(o in needed for o in n.output):keep.append(n);needed.update(n.input)
del m.graph.node[:];m.graph.node.extend(reversed(keep))
inits=[i for i in m.graph.initializer if i.name in needed];del m.graph.initializer[:];m.graph.initializer.extend(inits)
inputs=[i for i in m.graph.input if i.name in needed];del m.graph.input[:];m.graph.input.extend(inputs)
del m.graph.value_info[:]
onnx.checker.check_model(m);a.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,a.output);print('replaced',count,'remaining_nodes',len(m.graph.node))

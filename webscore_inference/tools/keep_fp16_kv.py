"""Remove lossless FP16→FP32→FP16 KV roundtrip casts in a matched feedback export."""
import argparse
from pathlib import Path
import onnx
from onnx import TensorProto as T

def change(graph):
    victims=[]
    substitutions={}
    for state in 'kv':
        past=f'past_{state}'
        present=f'present_{state}'
        vi_in=next(v for v in graph.input if v.name==past)
        vi_out=next(v for v in graph.output if v.name==present)
        if tuple(v.type.tensor_type.elem_type for v in (vi_in, vi_out))!=(T.FLOAT,T.FLOAT):
            raise ValueError('Expected existing FP32 state I/O')
        consuming=[n for n in graph.node if past in n.input]
        producing=[n for n in graph.node if present in n.output]
        if len(consuming)!=1 or len(producing)!=1: raise ValueError('Expected one wrapping cast for each state')
        before,after=consuming[0],producing[0]
        for node,typ in [(before,T.FLOAT16),(after,T.FLOAT)]:
            if node.op_type!='Cast' or list(node.attribute)[0].i!=typ or len(node.input)!=1 or len(node.output)!=1:
                raise ValueError('Did not recognize KV boundary cast')
        # The FP16 internal output is type-checked explicitly after conversion.
        substitutions[before.output[0]]=past
        substitutions[after.input[0]]=present
        vi_in.type.tensor_type.elem_type=T.FLOAT16
        vi_out.type.tensor_type.elem_type=T.FLOAT16
        victims += [before.name,after.name]
    for n in list(graph.node):
        if n.name in victims: graph.node.remove(n);continue
        for i,x in enumerate(n.input): n.input[i]=substitutions.get(x,x)
        for i,x in enumerate(n.output): n.output[i]=substitutions.get(x,x)
    for vi in list(graph.value_info):
        if vi.name in substitutions: graph.value_info.remove(vi)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
    if args.source.resolve()==args.output.resolve(): raise ValueError('Output must not overwrite the source graph')
    model=onnx.load(args.source)
    change(model.graph)
    onnx.checker.check_model(model,full_check=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(model,args.output)

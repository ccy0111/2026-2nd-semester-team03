import json,copy
from pathlib import Path
import onnx,numpy as np
from onnx import helper as h, numpy_helper as nh, TensorProto as T,compose
import argparse
parser=argparse.ArgumentParser(description="Build opt-in GPU token-feedback graph from standard exports")
parser.add_argument('--source-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();source=args.source_dir
if args.output.resolve() in {(source/n).resolve() for n in ['token_embed.onnx','decoder_kv.onnx','mel_encoder.onnx','config.json','tokenizer.json']}:
 raise ValueError('Output must not overwrite source assets')
args.output.parent.mkdir(parents=True,exist_ok=True)
e=compose.add_prefix(onnx.load(source/'token_embed.onnx'),'emb/');d=compose.add_prefix(onnx.load(source/'decoder_kv.onnx'),'dec/')
dim=next(v for v in e.graph.output if v.name=='emb/token_embeds').type.tensor_type.shape.dim[2].dim_value
dec_dim=next(v for v in d.graph.input if v.name=='dec/inputs_embeds').type.tensor_type.shape.dim[2].dim_value
if dim <= 0 or dim != dec_dim: raise ValueError('Embedding/decoder dimensions must match and be fixed')
# Preserve existing converted arithmetic and cast boundaries exactly.
nodes=list(e.graph.node)+[h.make_node('Concat',['prefix_embeds','emb/token_embeds'],['dec/inputs_embeds'],axis=1)]+list(d.graph.node)
tok=json.loads((source/'tokenizer.json').read_text());vocab=len(tok['id_to_token']);ban=np.zeros(vocab,np.float32);ban[[tok['pad_id'],tok['bos_id'],tok['unk_id']]]=-np.inf
nodes += [h.make_node('Gather',['dec/logits','last_index'],['last_logits'],axis=1),h.make_node('Add',['last_logits','blocked_tokens'],['allowed_logits']),h.make_node('ArgMax',['allowed_logits'],['next_token'],axis=-1,keepdims=0,select_last_index=0)]
inputs=[h.make_tensor_value_info('prefix_embeds',T.FLOAT,['batch','prefix_len',dim])]+list(e.graph.input)+[x for x in d.graph.input if x.name!='dec/inputs_embeds']
outputs=[h.make_tensor_value_info('next_token',T.INT64,['batch'])]+[x for x in d.graph.output if x.name!='dec/logits']
g=h.make_graph(nodes,'webscore-fused-greedy',inputs,outputs,initializer=list(e.graph.initializer)+list(d.graph.initializer)+[nh.from_array(np.array(-1,np.int64),'last_index'),nh.from_array(ban,'blocked_tokens')])
names={'emb/input_ids':'input_ids',**{'dec/'+n:n for n in ['past_k','past_v','position_offset','attn_mask','present_k','present_v']}}
for node in g.node:
 for field in (node.input,node.output):
  for i,name in enumerate(field):field[i]=names.get(name,name)
for field in (g.input,g.output,g.value_info):
 for value in field:value.name=names.get(value.name,value.name)
m=h.make_model(g,opset_imports=list(d.opset_import),ir_version=d.ir_version);
# Move last-position selection before vocabulary projection (not before attention/KV).
producers={o:n for n in m.graph.node for o in n.output}
producer=producers['dec/logits']
while producer.op_type=='Cast':producer=producers[producer.input[0]]
if producer.op_type!='MatMul':raise ValueError('Expected final vocabulary MatMul')
new_nodes=[]
for node in m.graph.node:
 if node is producer or node.output==producer.output:
  new_nodes.append(h.make_node('Gather',[node.input[0],'last_index'],['last_hidden'],axis=1));node.input[0]='last_hidden'
 if list(node.output)==['last_logits']:continue
 for i,n in enumerate(node.input):
  if n=='last_logits':node.input[i]='dec/logits'
 new_nodes.append(node)
del m.graph.node[:];m.graph.node.extend(new_nodes)
m.graph.input.append(h.make_tensor_value_info('token_history',T.INT64,['batch','history_len']))
m.graph.initializer.append(nh.from_array(np.array([1],np.int64),'feedback_axes'))
m.graph.node.extend([h.make_node('Unsqueeze',['next_token','feedback_axes'],['next_ids']),h.make_node('Concat',['token_history','next_ids'],['updated_history'],axis=1)])
outputs=[x for x in m.graph.output if x.name!='next_token'];del m.graph.output[:];m.graph.output.extend(outputs+[h.make_tensor_value_info('next_ids',T.INT64,['batch',1]),h.make_tensor_value_info('updated_history',T.INT64,['batch','updated_history_len'])])
onnx.checker.check_model(m);onnx.save(m,args.output)
print(args.output)

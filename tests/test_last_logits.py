"""Optional real ONNX numerical checks (requires onnx + onnxruntime)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_ONNX=importlib.util.find_spec('onnx') is not None and importlib.util.find_spec('onnxruntime') is not None

@unittest.skipUnless(HAS_ONNX, 'install onnx and onnxruntime for export tests')
class LastLogitsTests(unittest.TestCase):
 def test_optimized_matches_last_position_and_never_overwrites(self):
  import numpy as np
  import onnx
  import onnxruntime as ort
  from onnx import helper as h,numpy_helper as nh,TensorProto as T
  spec=importlib.util.spec_from_file_location('optimizer',Path(__file__).parents[1]/'webscore_inference/tools/optimize_last_logits.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  with tempfile.TemporaryDirectory() as td:
   src=Path(td)/'source';src.mkdir();dst=Path(td)/'optimized'
   nodes=[h.make_node('LayerNormalization',['x','scale','bias'],['norm'],axis=-1),h.make_node('MatMul',['norm','weight'],['logits'])]
   inits=[nh.from_array(np.ones(4,np.float32),'scale'),nh.from_array(np.zeros(4,np.float32),'bias'),nh.from_array(np.arange(12,dtype=np.float32).reshape(4,3),'weight')]
   m=h.make_model(h.make_graph(nodes,'last',[h.make_tensor_value_info('x',T.FLOAT,[1,'seq',4])],[h.make_tensor_value_info('logits',T.FLOAT,[1,'seq',3])],inits),opset_imports=[h.make_opsetid('',17)]);m.ir_version=10
   onnx.save(m,src/'decoder_kv.onnx')
   for name in ['config.json','tokenizer.json']: (src/name).write_text('{}')
   for name in ['mel_encoder.onnx','token_embed.onnx']: onnx.save(m,src/name)
   mod.optimize(src,dst,quantize=False)
   a=ort.InferenceSession(str(src/'decoder_kv.onnx'));b=ort.InferenceSession(str(dst/'decoder_kv.onnx'))
   for seq in [1,4,501]:
    x=np.random.default_rng(seq).normal(size=(1,seq,4)).astype(np.float32)
    np.testing.assert_allclose(b.run(None,{'x':x})[0],a.run(None,{'x':x})[0][:,-1:,:],rtol=1e-6,atol=1e-6)
   with self.assertRaises(ValueError):mod.optimize(src,dst,quantize=False)
   self.assertEqual(json.loads((dst/'config.json').read_text())['decoder_logits'],'last')

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree
import numpy as np
from backend.transcription import ModelLoadError, UnsupportedAudioFormatError, InvalidAudioError, InferenceError
from backend.webscore_cpu import MelFrontend, WebScoreTranscriber, read_metadata, decode_tokens, decode_mp3
from backend.webscore_score import render_score

SOURCE = Path(__file__).resolve().parents[1] / 'webscore_inference'

class AdapterTests(unittest.TestCase):
    def metadata(self):
        return json.loads((SOURCE/'config.json').read_text(encoding='utf-8-sig')), json.loads((SOURCE/'tokenizer.json').read_text(encoding='utf-8-sig'))

    def test_metadata(self):
        cfg, tok = read_metadata(SOURCE)
        self.assertEqual(cfg['card'], len(tok['id_to_token']))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with self.assertRaises(ModelLoadError): read_metadata(p)
            (p/'config.json').write_text(json.dumps(cfg))
            tok['id_to_token'][5] = {}
            (p/'tokenizer.json').write_text(json.dumps(tok))
            with self.assertRaises(ModelLoadError): read_metadata(p)

    def test_frontend(self):
        x=MelFrontend().compute(np.zeros(80000,dtype=np.float32))
        self.assertEqual(x.shape,(1,501,512))
        self.assertEqual(x.dtype,np.float32)
        np.testing.assert_allclose(x,np.log(1e-6),atol=1e-6)
        self.assertTrue(np.isfinite(MelFrontend().compute(np.ones(400))).all())

    def test_ties(self):
        _,tok=self.metadata(); labels=tok['id_to_token']; ids=lambda a:[labels.index(x) for x in a]
        active={}; notes=[]
        decode_tokens(ids(['inst:0','vel:1','pitch:60','<eos>']),labels,0,active,notes)
        decode_tokens(ids(['inst:0','pitch:60','tie','time:100','vel:0','pitch:60','<eos>']),labels,5,active,notes)
        self.assertEqual(notes,[(0,60,0.,6.)]); self.assertFalse(active)

    def test_svg(self):
        svg=render_score([(0,60,0,3),(40,68,1,2)])
        root=ElementTree.fromstring(svg)
        self.assertTrue(root.tag.endswith('svg')); self.assertIn(b'ellipse',svg)
        self.assertIn(b'No notes',render_score([]))
        extreme=ElementTree.fromstring(render_score([(0,0,0,1),(0,127,0,1),(1,60,0,1)]))
        height=float(extreme.attrib['height'])
        for node in extreme.iter():
            if node.tag.endswith('ellipse'):
                self.assertGreater(float(node.attrib['cy']),70)
                self.assertLess(float(node.attrib['cy']),height-10)

    def test_audio_errors(self):
        class Probe:
            stdout=b'{"format":{"format_name":"wav"},"streams":[{"codec_type":"audio","codec_name":"pcm_s16le"}]}'
        with patch('backend.webscore_cpu.subprocess.run',return_value=Probe()) as run:
            with self.assertRaises(UnsupportedAudioFormatError):decode_mp3(Path('test.mp3'),'probe','ffmpeg')
            self.assertIn('-protocol_whitelist',run.call_args.args[0])
        with patch('backend.webscore_cpu.subprocess.run',side_effect=OSError()):
            with self.assertRaises(InvalidAudioError):decode_mp3(Path('test.mp3'),'probe','ffmpeg')

    def test_mock_onnx(self):
        cfg,tok=self.metadata(); cfg=dict(cfg,dim=8,num_layers=1,num_heads=2,num_kv_heads=1)
        labels=tok['id_to_token']; output=iter([labels.index(s) for s in ['vel:1','pitch:60','time:100','vel:0','pitch:60','<eos>']])
        class Encoder:
            def run(self,names,feeds):
                assert feeds['mel'].shape==(1,501,512)
                return [np.zeros((1,2,8),np.float32)]
        class Embed:
            def run(self,names,feeds):
                assert feeds['input_ids'].dtype==np.int64
                return [np.zeros((1,feeds['input_ids'].shape[1],8),np.float32)]
        class Decoder:
            def run(self,names,feeds):
                seq=feeds['inputs_embeds'].shape[1]; past=feeds['past_k'].shape[3]
                assert feeds['attn_mask'].shape==(1,1,seq,past+seq)
                assert feeds['position_offset'][0]==past
                logits=np.zeros((1,seq,len(labels)),np.float32); logits[0,-1,next(output)]=1
                kv=np.zeros((1,1,1,past+seq,4),np.float32)
                return [logits,kv,kv]
        model=WebScoreTranscriber(cfg,tok,[Encoder(),Embed(),Decoder()],'probe','ffmpeg')
        with patch('backend.webscore_cpu.decode_mp3',return_value=np.zeros(32000,np.float32)):
            score=model.transcribe(Path('test.mp3'))
        self.assertEqual(score.media_type,'image/svg+xml'); ElementTree.fromstring(score.content)

    def test_no_eos(self):
        cfg,tok=self.metadata()
        model=WebScoreTranscriber(cfg,tok,[None,None,None],'probe','ffmpeg')
        with patch('backend.webscore_cpu.decode_mp3',return_value=np.zeros(1)),patch.object(model,'generate',side_effect=RuntimeError('failed')):
            with self.assertRaises(InferenceError):model.transcribe(Path('test.mp3'))

if __name__=='__main__':unittest.main()

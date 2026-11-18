"""Server manifest adapter security/contract tests; no cloud calls."""
import io
import json
import unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.model_manifest import (ManifestSettings, ModelManifestService,
                                    create_model_manifest_router, validate_catalog)

SETTINGS = ManifestSettings(account_id='a'*32, bucket='midi-course-checkpoints')

def catalog():
    names=('config.json','tokenizer.json','mel_encoder.onnx','token_embed.onnx','decoder_kv.int8.onnx')
    return dict(schemaVersion=1, format='webscore-onnx-v1', version='v1', precision='int8',
                files={n:dict(key=f'web-student-kd/releases/v1/{n}',size=123,sha256='a'*64) for n in names})

class FakeS3:
    def __init__(self): self.catalog=catalog(); self.calls=[]; self.fail=False
    def get_object(self,**kwargs):
        self.calls.append(('get',kwargs))
        if self.fail: raise RuntimeError('https://secret?signature=LEAK')
        self.body=io.BytesIO(json.dumps(self.catalog).encode())
        return {'Body':self.body}
    def generate_presigned_url(self,operation,**kwargs):
        self.calls.append(('sign',operation,kwargs))
        return 'https://example.test/'+kwargs['Params']['Key']+'?signature=TEST_ONLY'

class ManifestTests(unittest.TestCase):
    def setUp(self): self.s3=FakeS3(); self.service=ModelManifestService(SETTINGS,self.s3)
    def client(self, authorize):
        app=FastAPI();app.include_router(create_model_manifest_router(self.service,authorize));return TestClient(app)
    def test_explicit_auth_required_and_fail_closed(self):
        with self.assertRaises(TypeError):create_model_manifest_router(self.service,None)
        for result in (False,None,'yes',1):
            response=self.client(lambda request:result).get('/api/model-manifest')
            self.assertEqual(response.status_code,403);self.assertIn('no-store',response.headers['cache-control'])
        self.assertEqual(self.s3.calls,[])
    def test_auth_exception_redacted(self):
        def deny(request):raise RuntimeError('secret')
        response=self.client(deny).get('/api/model-manifest')
        self.assertEqual(response.status_code,403);self.assertNotIn('secret',response.text)
    def test_default_app_has_no_manifest_route(self):
        from backend.main import create_app
        self.assertNotIn('/api/model-manifest',[getattr(r,'path',None) for r in create_app().routes])
    def test_authorized_response_fixed_objects_and_no_store(self):
        response=self.client(lambda request:True).get('/api/model-manifest?key=other&bucket=other')
        self.assertEqual(response.status_code,200);self.assertEqual(response.json()['version'],'v1')
        self.assertEqual(len(response.json()['files']),5);self.assertIn('no-store',response.headers['cache-control'])
        self.assertEqual(self.s3.calls[0][1],dict(Bucket=SETTINGS.bucket,Key=SETTINGS.catalog_key))
        for _,op,args in self.s3.calls[1:]:
            self.assertEqual(op,'get_object');self.assertEqual(args['HttpMethod'],'GET');self.assertEqual(args['ExpiresIn'],3600)
            self.assertTrue(args['Params']['Key'].startswith('web-student-kd/releases/v1/'))
        self.assertTrue(self.s3.body.closed)
    def test_async_authorization_supported(self):
        async def allow(request):return True
        self.assertEqual(self.client(allow).get('/api/model-manifest').status_code,200)
    def test_catalog_snapshot_renewal_and_rollover(self):
        self.service.runtime_manifest();self.s3.catalog['version']='v2'
        for item in self.s3.catalog['files'].values():item['key']=item['key'].replace('/v1/','/v2/')
        self.assertEqual(self.service.runtime_manifest()['version'],'v2')
    def test_bad_catalog_rejected_before_any_signing(self):
        mutations=[lambda c:c.update(schemaVersion=2),lambda c:c['files'].pop('config.json'),
                   lambda c:c['files']['config.json'].update(key='another/private/file'),
                   lambda c:c['files']['config.json'].update(key='https://evil.example/model'),
                   lambda c:c['files']['config.json'].update(size=True),
                   lambda c:c['files']['config.json'].update(sha256='bad'),
                   lambda c:c.update(version='../other')]
        for mutate in mutations:
            self.s3.calls=[];self.s3.catalog=catalog();mutate(self.s3.catalog)
            response=self.client(lambda request:True).get('/api/model-manifest')
            self.assertEqual(response.status_code,503);self.assertEqual(len(self.s3.calls),1)
    def test_provider_error_redacted(self):
        self.s3.fail=True;response=self.client(lambda request:True).get('/api/model-manifest')
        self.assertEqual(response.status_code,503);self.assertNotIn('signature',response.text)
    def test_catalog_size_bound(self):
        self.s3.catalog['padding']='x'*65536
        with self.assertRaises(ValueError):self.service.runtime_manifest()
        self.assertTrue(self.s3.body.closed)
    def test_settings_do_not_repr_credentials_and_invalid_paths_rejected(self):
        settings=ManifestSettings(account_id='a'*32,bucket='abc',access_key_id='private-id',secret_access_key='private-secret')
        self.assertNotIn('private-',repr(settings))
        with self.assertRaises(ValueError):ManifestSettings(account_id='wrong',bucket='abc')
        with self.assertRaises(ValueError):ManifestSettings(account_id='a'*32,bucket='abc',release_prefix='../oops')
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(ValueError,'settings'):ManifestSettings.from_env()

if __name__=='__main__': unittest.main()

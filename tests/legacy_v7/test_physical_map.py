import json, struct, tempfile, unittest
from pathlib import Path
from gguf_utils import parse_gguf

class GGUFMapTests(unittest.TestCase):
    def test_minimal_gguf_tensor_index(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.gguf'
            meta={'general.alignment':32,'general.architecture':'test','test.expert_count':2,'test.block_count':1}
            b=bytearray(b'GGUF'); b += struct.pack('<IQQ',3,1,len(meta))
            def s(x):
                raw=x.encode(); return struct.pack('<Q',len(raw))+raw
            for k,v in meta.items():
                b+=s(k)
                if isinstance(v,str): b+=struct.pack('<I',8)+s(v)
                elif isinstance(v,int): b+=struct.pack('<I',4)+struct.pack('<I',v)
            b+=s('blk.0.ffn_up_exps.weight')+struct.pack('<I',3)+struct.pack('<QQ',4,2)+struct.pack('<I',1)+struct.pack('<Q',0)
            while len(b)%32:b+=b'\0'
            b+=b'\0'*16
            p.write_bytes(b)
            g=parse_gguf(p)
            self.assertEqual(g['metadata']['test.expert_count'],2)
            self.assertEqual(g['tensors'][0].name,'blk.0.ffn_up_exps.weight')
            self.assertEqual(g['data_start'] % 32, 0)

if __name__=='__main__':unittest.main()

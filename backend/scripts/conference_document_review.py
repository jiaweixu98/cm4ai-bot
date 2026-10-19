"""Small public proposal fixtures through the actual local Next document endpoint."""
import base64
import io
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import zipfile

text='Proposed paper: pathway enrichment and gene-expression networks for disease-associated molecular phenotypes.'
for attempt in range(40):
    try:
        with urlopen('http://127.0.0.1:8100/api/health',timeout=1) as health:
            if health.status==200:break
    except (URLError,TimeoutError):
        pass
    time.sleep(0.5)
else:
    raise RuntimeError('Local backend is not ready')
stream=f'BT /F1 12 Tf 40 750 Td ({text}) Tj ET'.encode()
objects=[b'<< /Type /Catalog /Pages 2 0 R >>',b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
 b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 900 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
 b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
 b'<< /Length '+str(len(stream)).encode()+b' >>\nstream\n'+stream+b'\nendstream']
pdf=b'%PDF-1.4\n';offsets=[]
for i,obj in enumerate(objects,1):
    offsets.append(len(pdf));pdf+=str(i).encode()+b' 0 obj\n'+obj+b'\nendobj\n'
start=len(pdf)
pdf+=b'xref\n0 6\n0000000000 65535 f \n'+b''.join(f'{offset:010d} 00000 n \n'.encode() for offset in offsets)
pdf+=b'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n'+str(start).encode()+b'\n%%EOF'
word=io.BytesIO()
with zipfile.ZipFile(word,'w') as archive:
    archive.writestr('word/document.xml',f'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>')
cases=[]
for name,data,status in [('proposal.pdf',pdf,200),('proposal.docx',word.getvalue(),200),('unreadable.docx',b'not a zip document',422)]:
    request=Request('http://127.0.0.1:3100/api/attachment-text',data=json.dumps({
      'filename':name,'data_base64':base64.b64encode(data).decode()}).encode(),headers={'Content-Type':'application/json'})
    started=time.monotonic()
    try:
        with urlopen(request,timeout=150) as response:
            code=response.status;payload=json.load(response)
    except HTTPError as error:
        code=error.code;payload=json.load(error)
    assert code==status,(name,code)
    if status==200:
        assert 'pathway' in payload['text'].lower() and 'phenotype' in payload['text'].lower()
    cases.append({'filename':name,'status':code,'seconds':round(time.monotonic()-started,1),'result':payload})
    print(name,code,flush=True)
Path('/tmp/conference-review-20261016/documents.json').write_text(json.dumps({'complete':True,'cases':cases},indent=2))

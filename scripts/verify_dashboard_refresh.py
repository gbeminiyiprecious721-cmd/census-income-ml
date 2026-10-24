"""Optional idle-browser verification of the actual 60-second refresh fragment."""
import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
from streamlit.proto.BackMsg_pb2 import BackMsg
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
 page=b.new_page(viewport={'width':1440,'height':900});frames=[]
 def sent(payload):
  if isinstance(payload,bytes):
   m=BackMsg();m.ParseFromString(payload)
   if m.WhichOneof('type')=='rerun_script':frames.append({'time':time.monotonic(),'fragment_id':m.rerun_script.fragment_id})
 page.on('websocket',lambda ws:ws.on('framesent',sent))
 page.goto('http://127.0.0.1:8501');page.get_by_role('heading',name='Prediction distribution',exact=True).wait_for(timeout=90000)
 start=time.monotonic();page.wait_for_timeout(65000)
 automatic=[{'seconds_after_ready':round(x['time']-start,2),'fragment_id':x['fragment_id']} for x in frames if x['time']>start and x['fragment_id']]
 assert automatic,frames
 (Path(__file__).resolve().parents[1]/'docs/premium_dashboard/refresh_qa.json').write_text(json.dumps({'automatic_fragment_requests':automatic,'observation_seconds':65},indent=2)+'\n')
 print(automatic);b.close()

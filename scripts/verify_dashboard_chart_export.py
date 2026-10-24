"""Verify the Plotly PNG download and collect browser console/HTTP failures."""
import json,re
from pathlib import Path
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True)
 page=browser.new_page(viewport={'width':1440,'height':900});console=[];http=[]
 page.on('console',lambda m:console.append(m.text) if m.type=='error' else None)
 page.on('response',lambda r:http.append({'status':r.status,'url':r.url}) if r.status>=400 else None)
 page.goto('http://127.0.0.1:8501');page.get_by_role('heading',name='Prediction distribution',exact=True).wait_for(timeout=90000)
 page.locator('.st-key-overview_cv .main-svg').first.wait_for(timeout=90000)
 page.locator('.st-key-overview_cv').hover()
 with page.expect_download(timeout=30000) as event:
  page.locator('.st-key-overview_cv [data-title="Download plot as a PNG"]').click()
 raw=Path(event.value.path()).read_bytes();assert raw.startswith(b'\x89PNG\r\n\x1a\n')
 for name in ['Data Explorer','Model Performance','Model Explainability','Responsible AI Assessment','Prediction Sandbox']:
  page.get_by_role('radio',name=re.compile(name)).focus();page.keyboard.press('Space');page.wait_for_timeout(500)
  page.wait_for_function("document.querySelector('[data-testid=\"stApp\"]').getAttribute('data-test-script-state')==='notRunning'",timeout=90000)
 assert not console,console
 assert not http,http
 out=Path(__file__).resolve().parents[1]/'docs/premium_dashboard/chart_export_qa.json'
 out.write_text(json.dumps({'png_bytes':len(raw),'console_errors':console,'http_errors':http},indent=2)+'\n');print(out.read_text());browser.close()

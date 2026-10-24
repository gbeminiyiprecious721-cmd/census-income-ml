from playwright.sync_api import sync_playwright
import json
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless=True);page=b.new_page(viewport={'width':1440,'height':900});page.goto('http://127.0.0.1:8501');page.get_by_role('heading',name='Prediction distribution',exact=True).wait_for(timeout=90000)
 def ready():page.wait_for_timeout(600);page.wait_for_function("document.querySelector('[data-testid=\"stApp\"]').getAttribute('data-test-script-state')==='notRunning'",timeout=90000)
 def export():
  with page.expect_download() as d:page.get_by_role('button',name='Export results',exact=True).click()
  return json.load(open(d.value.path()))['context']
 ready();slider=page.locator('.st-key-analysis_threshold').get_by_role('slider');slider.focus();page.keyboard.press('ArrowRight');ready();print('changed',export(),flush=True)
 page.get_by_role('button',name='Reset filters',exact=True).click();ready();print('reset',export(),flush=True)
 page.get_by_role('button',name='Refresh analysis',exact=True).click();ready();current=export();print('refresh',current,flush=True);assert current['threshold']==.5
 page.get_by_role('radio',name='06   Prediction Sandbox').focus();page.keyboard.press('Space');ready();print('sandbox',page.locator('.st-key-sandbox_threshold').inner_text(),flush=True);b.close()
